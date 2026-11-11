# Abyss manual

How to build Abyss, set up DeepSeek V4 Flash (and V4.1 Flash), run it and tune it. Japanese: [MANUAL.ja.md](MANUAL.ja.md).

Abyss is a fork of [Project Maya](https://github.com/mw00/project-maya) (MIT), itself grown out of
[Strata](https://github.com/Niko1221/Strata) (MIT). The engine, the expert tiers, the server and the dashboard are
Maya's; DeepSeek V4 and V4.1 support is what Abyss adds. Maya's own documentation (GLM-5.3-Flash, AMD, multi-GPU, the
installer) is kept under [docs/maya/](maya/README.md).

- [What you need](#what-you-need)
- [Setup in one command](#setup-in-one-command)
- [Build](#build)
- [Get the model](#get-the-model)
- [Make the pack](#make-the-pack)
- [Configure and start](#configure-and-start)
- [Using it](#using-it)
- [DeepSeek V4.1 Flash](#deepseek-v41-flash)
- [Qwen3.8-Flash-Next](#qwen38-flash-next)
- [Ornith-1.5-122B](#ornith-15-122b)
- [EXL3 quants (GLM-5.3-Flash)](#exl3-quants-glm-53-flash)
- [Memory and speed](#memory-and-speed)
- [The DSpark drafter (DeepSeek V4 Flash)](#the-dspark-drafter-deepseek-v4-flash)
- [Uncensoring at run time (`--uncensor`)](#uncensoring-at-run-time---uncensor)
- [Settings](#settings)
- [Checking the output against llama.cpp](#checking-the-output-against-llamacpp)
- [Something went wrong?](#something-went-wrong)
- [Limitations](#limitations)

## What you need

| | Tested | Notes |
| --- | --- | --- |
| OS | Ubuntu (Linux) | Windows 11: run by users (`setup-deepseek.bat`; RTX 4090 #31, RX 7900 XT #5), not tested here |
| GPU | RTX 5090 (32 GB), RTX 3090 | build for your GPU's architecture (`120a` for RTX 50, `86` for RTX 30; several at once for a split). On pre-Blackwell GPUs the MXFP4 expert rows run through the q8_1 path instead of FP4 MMA; before Ampere (sm_80) the indexer and the prompt attention use the older FP32 kernels. A V100 (sm_70) builds with a CUDA 12 toolkit but has not been run. AMD Radeon: see "AMD GPUs" (experimental) |
| RAM | 94 GB | the model is 96.8 GB: with VRAM + RAM above ~100 GB every expert stays in memory; with less, the rest are read from the SSD (works, slower) |
| Disk | NVMe SSD, ~100 GB free | the GGUF files; the pack adds 0.2 GB |
| CUDA | 13.3 | with `nvcc`; 12.9 for a V100 (`--nvcc`) |
| Compiler | g++ 13 | a version your CUDA accepts |
| Python | 3.12 | a private `.venv` in the Abyss folder |
| ROCm (optional) | 10 (HIP 7.14 or newer, `hipcc`) | an AMD APU's integrated GPU computes the RAM-tier experts (the iGPU lane, below); without it the CPU does |
| ARM (beta) | NVIDIA DGX Spark (GB10, aarch64) | the engine built by hand for `121a` with CUDA 12.9 or newer; see "NVIDIA DGX Spark" below |

`./abyss.sh` (Maya's installer, renamed) still installs Maya's GLM-5.3-Flash models. DeepSeek V4 is set up with
`./setup-deepseek.sh` (next section), or by hand with the steps after it.

## Setup in one command

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model
```

DeepSeek V4.1 Flash 1017 with zephel01's LDLQ GGUF, made for a 32 GB GPU + 96 GB RAM (BETA; `--quant 139G` for the
quality build, 132G by default for speed; [the model card](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF)):

```sh
./setup-deepseek.sh --model v41 --download-model
```

The smaller, general-purpose [Compact JA256](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF) (207 GB, 256 of 384 experts, Japanese intact, ~30% faster on
32 GB + 96 GB) is not in the setup yet: download its `JA256/` folder and pack it by hand as in
[docs/deepseek41/README.md](deepseek41/README.md#ggufs-that-load).

It writes `abyss-dsv41-132g.json` with the card's everyday settings (`STRATA_GLM_TAIL_SKIP=4`,
`STRATA_GLM_TAIL_SKIP_DISK=1`, `STRATA_GLM_ROUTE_AFFINITY_RAM=0.08`, 8 disk readers); the DSpark drafter is not set up
for V4.1 (by hand: [docs/deepseek41](deepseek41/README.md)). Other V4.1 GGUFs are set up by hand as described there.

The two Qwen3.5-MoE 122B models in zephel01's quants for a 16 GB GPU + 64 GB RAM (51-52 GB; Ornith drafts with the
NextN block in its second shard):

```sh
./setup-deepseek.sh --model ornith --download-model     # Ornith-1.5-122B-IQ3_XXS-16G64G
./setup-deepseek.sh --model qwen35 --download-model     # Qwen3.5-122B-A10B-IQ3_XXS-16G64G
```

zephel01's other GGUFs, one repo per PC size: `--quant` names the size (the RAM tier in the config is the one the model
card was measured with, from this PC's RAM; Qwen3.8 drafts with the MTP block in its shards; the DSpark drafter is left
off for the V4 sizes, which run faster without it on a discrete GPU):

```sh
./setup-deepseek.sh --model ornith --quant 24G64G --download-model   # Ornith-1.5-122B-IQ4_XS-24G64G (68.9 GB)
./setup-deepseek.sh --model qwen38 --quant 12G32G --download-model   # Qwen3.8-Flash-Next-E448-IQ2_M-12G32G (67.5 GB)
./setup-deepseek.sh --model qwen38 --download-model                  # Qwen3.8-Flash-Next-IQ3_M-16G64G (86.9 GB)
./setup-deepseek.sh --model qwen38 --quant 24G64G --download-model   # Qwen3.8-Flash-Next-IQ3_M-24G64G (93.7 GB)
./setup-deepseek.sh --model glm --download-model                     # GLM-5.3-Flash-IQ2_XS-32G96G (94.1 GB)
./setup-deepseek.sh --model glm --quant 32G128G --download-model     # GLM-5.3-Flash-IQ3_XXS-32G128G (127.9 GB)
./setup-deepseek.sh --quant 16G64G --download-model                  # DeepSeek-V4-Flash-0731-IQ1_S-16G64G (61.7 GB)
./setup-deepseek.sh --quant 32G64G --download-model                  # DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G (72.0 GB)
./setup-deepseek.sh --quant 32G96G --download-model                  # DeepSeek-V4-Flash-0731-IQ2_M-32G96G (96.8 GB)
```

`setup-deepseek.sh` makes the `.venv`, then runs `tools/abyss_setup.py`. Each step is skipped when it is already done,
so it can be run again at any time:

1. checks the PC (an NVIDIA GPU, RAM);
2. Python packages, llama.cpp's source at the pinned commit;
3. compiles the engine for the GPU (`120a` on RTX 50);
4. downloads the model to `../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<quant>` - it asks first unless
   `--download-model` is given - and checks every file against Hugging Face's SHA-256;
5. makes the pack next to the shards;
6. writes the run config `abyss-<name>.json` and the start script `run-abyss-<name>.sh` in the Abyss folder;
7. the DSpark drafter (v1.3.43, optional): when the config is new it asks whether to download DeepSeek's DSpark
   drafter (6.9 GB) and turn it on - `--dspark` without asking - then checks its SHA-256 and names it in the config's
   `env` ([The DSpark drafter](#the-dspark-drafter-deepseek-v4-flash));
8. creates the **Abyss-tools** folder beside Abyss: `abyss.sh start | stop | restart | status | update | build | log |
   models`, from the templates in `ops/abyss-tools/`. Running the setup again brings `abyss.sh` / `common.sh` up to date
   with the templates (the old ones kept as `.bak`) and keeps your own `README.md` there.

```sh
../Abyss-tools/abyss.sh start      # then http://127.0.0.1:8082/
../Abyss-tools/abyss.sh stop
```

| Option | |
| --- | --- |
| `--quant UD-IQ3_XXS` | another quant of unsloth/DeepSeek-V4-Flash-0731-GGUF (default `UD-Q2_K_XL`, the tested one) |
| `--name NAME` | the model's name in `abyss-<name>.json` (default `dsv4-<quant>`, e.g. `dsv4-q2_k_xl`) |
| `--context N`, `--port N` | the context (default 65536) and the port (default 8082) |
| `--data-dir DIR`, `--tools-dir DIR` | other places for Abyss-data and Abyss-tools |
| `--gpu N`, `--arch A` | the GPU, the CUDA architecture to compile for |
| `--nvcc PATH` | the CUDA compiler to build with (a CUDA 12 one for a V100) |
| `--gpus 0,1`, `--gpus all` | split the layers across several GPUs (see "Several GPUs"); builds for every GPU's architecture |
| `--download-model`, `--yes` | download without asking; the defaults for every question |
| `--rewrite-config` | write `abyss-<name>.json` again (otherwise an existing one is kept; a DSpark drafter it named stays on) |
| `--dspark`, `--no-dspark` | download DeepSeek's DSpark drafter (6.9 GB, to `../Abyss-data/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF`) and turn it on in the config; turn it off (the file stays). One GPU only |
| `--tools-only` | only create / update Abyss-tools |

The sections below are the same steps by hand.

## Build

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss

# the private Python environment
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil

# llama.cpp's source at the commit the engine is pinned to (setup.py LLAMA_CPP_COMMIT): ggml, the MMQ kernels, gguf-py
C=3cf03257f219afbe7334045ff7c6a06ac68c627d
curl -L -o /tmp/llama.zip https://github.com/ggml-org/llama.cpp/archive/$C.zip
unzip -q /tmp/llama.zip -d third_party && mv third_party/llama.cpp-$C third_party/llama.cpp

# the engine: build/strata
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
```

With ROCm's `hipcc` installed, the build also makes `build/libabyss_igpu.so` (the iGPU lane, below;
`-DSTRATA_IGPU_ARCH=gfx1151` is the default target, `-DSTRATA_IGPU_LANE=OFF` skips it).

The first build takes a few minutes; after a change only what changed is compiled again.

> [!IMPORTANT]
> RTX 50 (and the RTX PRO Blackwell cards): use `120a`, not `120`. The MXFP4 kernels use Blackwell's FP4 MMA
> instructions, which `120` cannot compile (`ptxas error : Instruction 'mma with block scale' not supported on .target
> 'sm_120'`, issue #40). DGX Spark: `121a`.

## Get the model

unsloth's [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF), quant
`UD-Q2_K_XL` (3 files, 96.8 GB), with the Hugging Face CLI (`pip install huggingface_hub` gives `hf`):

```sh
M=~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF   # Abyss-data beside Abyss (any NVMe folder works)
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf --local-dir $M
sha256sum $M/UD-Q2_K_XL/*.gguf
```

| File | SHA-256 |
| --- | --- |
| `-00001-of-00003.gguf` (5 MB) | `de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b` |
| `-00002-of-00003.gguf` (49.4 GB) | `c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327` |
| `-00003-of-00003.gguf` (47.4 GB) | `a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4` |

If a download stops and does not resume, `curl -L -C - -o <file> https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-Q2_K_XL/<file>`
continues it.

Other quants of the same repository should work if their expert types are among the decode kernels' (IQ1_S, IQ1_M,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K-Q6_K, MXFP4; IQ1_M has no prompt-path kernel, so such a quant reads
prompts token by token); only UD-Q2_K_XL is tested.

DeepSeek V4 GGUFs whose router and small projections are F16 rather than quantized (antirez's deepseek-v4-gguf, huihui-ai's
abliterated copies) also pack and run: `--compat-bf16` rounds those to BF16 for the engine's GEMVs. The huihui-ai Q2
file was checked against llama.cpp on the same file (KL 0.0075 over 3,000 tokens).

## Make the pack

The pack is the engine's index of the GGUF files: the small tensors (dense.bin), the expert table and the tokenizer
with the chat template. It goes next to the shards:

```sh
cd ~/llm/apps/Abyss
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL/pack-dsv4 --compat-bf16
```

`--compat-bf16` rounds the hyper-connection matrices (stored F32 in the GGUF) to BF16, the form the engine reads. A
few seconds; the experts stay in the GGUF files.

### A lighter dense pack (the attention projections at Q6_K or Q5_K)

Every decoded token reads all of the dense weights (6.6 GB in UD-Q2_K_XL), and 4.6 GB of that is three Q8_0 tensors a
layer: `attn_q_b`, `attn_output_a` and `attn_output_b`. `tools/requant_dense.py` re-quantizes just those (with
`ggml_quantize_chunk` from a llama.cpp build - llama-quantize refuses an i-quant model without an imatrix) and copies
every other byte of the shards as it is; `iq_pack.py` then packs the new files like the original ones:

```sh
cd ~/llm/apps/Abyss
GGUF_PY=~/llm/apps/llama.cpp/gguf-py LIBGGML=~/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so \
  .venv/bin/python -I tools/requant_dense.py $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  $M/UD-Q2_K_XL-attn-q6k q6_k 'blk\.\d+\.attn_q_b\.weight' 'blk\.\d+\.attn_output_[ab]\.weight'
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL-attn-q6k/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL-attn-q6k/pack-dsv4 --compat-bf16
```

It rewrites the 90 GB of shards (10-25 minutes on an NVMe), so run it while the server is stopped; then point the
config's pack and tokenizer paths at the new directory. What the three types cost and give (2026-10-15, v1.3.18;
KL against the engine's own Q8_0 reference logits over the last 1,000 of 4,000 tokens, CPU lane, the same setup for
all three):

| attention projections | the three tensors | KL vs reference | max KL | same top token | nll | VRAM expert slots (RTX 5090) |
|---|---|---|---|---|---|---|
| Q8_0 (the GGUF as published) | 4.28 GiB | 0.0060 | 0.11 | 97.0% | 1.3071 | 3,010 |
| Q6_K | 3.31 GiB | 0.0082 | 0.31 | 95.8% | 1.3116 | 3,139 |
| Q5_K | 2.77 GiB | 0.0119 | 0.57 | 95.9% | 1.3135 | 3,225 |
| Q4_K | 2.27 GiB | 0.0246 | 0.84 | 92.9% | 1.3224 | 3,268 |

- Q6_K: decode +3.5% from the smaller GEMVs (v1.3.18's split kernel) plus 129 more VRAM slots. On the swe-bench L7
  set (16 tasks x 3, RTX 5090 + Radeon 8060S) it ran 93.9 against 91.3 tok/s with the same resolved rate (81.2%) and
  quality score (81.2 vs 81.0). The KL cost is the run-to-run spread's size.
- Q5_K: the KL doubles against Q8_0 (the largest single-token divergence 0.57). On an RTX 5090 it decodes 4% faster
  than Q6_K (1,024-token answers: a single token 12.13 -> 11.65 ms, the verify window 15.52 -> 14.86, 86-89 -> 93
  tok/s; the window and the token path share the saving), and more on a card with less memory bandwidth - a Radeon AI
  PRO R9700 at 640 GB/s reads 1.5 GB less a token. On the same swe-bench set it ran 99.0 tok/s with resolved 87.5%
  (14/16), success rate 83.3% and quality 82.0 - no loss visible for twice Q6_K's KL. **This is the pack the author
  runs, and the one the decode figures under "Memory and speed" are measured with** (v1.3.20).
- Q4_K: where it stops paying. The KL doubles again and the top token changes 3 points more often (Q8_0 -> Q5_K
  cost 1.1), for a single token only 2% faster than Q5_K (11.67 -> 11.40 ms) and 43 more slots. Not run on swe-bench.
- Not worth touching: `attn_q_a`, the kv / compressor / indexer projections and the norms are small, and the shared
  experts, the embedding and the output head are already Q5_K / Q6_K / Q4_K.

## Configure and start

A config file (any name; `strata-dsv4.json` here) in the Abyss folder:

```json
{
 "exe": "/home/you/llm/apps/Abyss/build/strata",
 "args": ["--glm-pack", "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4",
          "--max-context", "65536", "--prefill", "auto"],
 "cwd": "/home/you/llm/apps/Abyss",
 "tokenizer": "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4/tokenizer",
 "model_name": "deepseek-v4-flash",
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082,
 "log": "/home/you/llm/apps/Abyss/strata-dsv4.log"
}
```

Every key, where setup puts this file (`abyss-<name>.json`) and how to change it: [SETTINGS.md](SETTINGS.md).

Start the server:

```sh
.venv/bin/python serve/server.py --engine strata --config strata-dsv4.json --port 8082
```

It prints `ready: http://127.0.0.1:8082/v1` after loading. A start takes about a minute: the dense weights go to the
GPU, then every expert is read into the VRAM and RAM tiers (about 20 s at 4 GB/s from an NVMe). Ctrl+C stops it.

Server options: `--host 0.0.0.0` to accept other machines (always with `--api-key <secret>`), `--open` opens the page
in the browser, `--gpu 0` picks the GPU.

## Using it

- **Browser:** `http://127.0.0.1:8082/` - chat, a live monitor of the expert tiers and the GPU, settings.
- **OpenAI-compatible API:** base URL `http://127.0.0.1:8082/v1`, any model name.

  ```sh
  curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
  ```

  The answer is in `content`, the thinking in `reasoning_content`.
- **Anthropic-compatible API:** `http://127.0.0.1:8082/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8082`; `./abyss.sh --connect claude` prints all its settings).
- **Thinking:** DeepSeek V4 thinks unless asked not to. `reasoning_effort` (or the chat's setting):

  | Level | What DeepSeek gets |
  | --- | --- |
  | not set | thinking on |
  | `none` | thinking off (the prompt ends with `</think>`) |
  | `low` | thinking on |
  | `high` | thinking on + DeepSeek's "Reasoning Effort: Absolute maximum" instruction |
  | `max` | thinking on + its "Beyond maximum" instruction |

  DeepSeek V4.1 1017's own template takes the level itself (v1.3.117): `low` / `medium` / `high` / `max` (its effort
  50 / 62 / 75 / 100) or an integer 1-100; `none` turns thinking off as above. Other models get the nearest level.

  The server always leaves the answer at least 1024 tokens (or a quarter of `max_tokens`): with a small `max_tokens`
  (a few hundred) the thinking is closed almost at once. Give thinking requests a `max_tokens` of a few thousand.
- **Tool calls:** OpenAI `tools` / Anthropic `tools` work, including several calls in one turn. The model writes them
  in DeepSeek's DSML form; the server turns them into `tool_calls` and renders past calls and `tool` results back into
  the prompt.
- **Conversation reuse:** a request that continues the previous one (the same messages plus new ones) reads only the
  new tokens; the same prompt again reads only its last one (V4.1: 5K and 23K tokens to the first token in 0.04 s).
  Only the last conversation is kept: another request in between replaces it, and so does a restart. The tokens must
  match from the start - a different system prompt, reasoning level or tool list, or anything that changes per
  request (a timestamp), reads the prompt again. Whether a request was reused: `usage.prompt_tokens_details.cached_tokens`
  (OpenAI), `cache_read_input_tokens` (Anthropic) or `timings.cache_n` in the stream. Ornith reuses it from v1.3.65.
- **One request at a time;** others wait in line.
- **Stopping the server from another program** (Abyss Terminal, a script, a phone over Tailscale when the engine runs
  away): `POST /api/shutdown` with a JSON body.

  ```sh
  curl -X POST http://127.0.0.1:8082/api/shutdown -H 'Content-Type: application/json' -d '{}'
  ```

  The request in progress is cut at its next step (not wrapped up as Ctrl+C does), and an engine that does not stop
  it within `STRATA_SHUTDOWN_WAIT_S` seconds (15) - stuck in a GPU kernel, say - is killed; then the engine ends and
  the server exits. `{"force": true}` kills the engine at once (its kept conversation is not saved). It answers
  `202` before it stops. Who may: with an `api_key` in the config, anyone who sends the key; without one, only
  programs on this PC (loopback) - another PC gets a `403`. A browser page cannot post it (it needs a JSON body and
  no foreign `Origin`). `"allow_shutdown": false` in the config turns it off. **Tailscale:** with `"host": "0.0.0.0"`
  (or the PC's Tailscale address) the other devices are not loopback, so give the server an `api_key` to stop it
  from your phone or laptop; behind `tailscale serve` (or any reverse proxy on this PC) every client looks like
  loopback - set an `api_key` there too, or anyone on your tailnet can stop it.
- **What `/metrics` reports for monitors** (the web app's Monitor, Abyss Terminal): the model's name with its quant
  (`engine.display_name`, `engine.quant`, `engine.config`, `engine.backend`); every card in `hardware.gpus` with its
  `role` (`main`, `split`, `helper`, `igpu` - the iGPU lane's Radeon, in shared memory) and `source` (the backends
  that answered: `nvml`, `nvidia-smi`, `sysfs`, `gpu_metrics`, `amd-smi`, `rocm-smi`, `pdh`), clocks, junction and
  memory temperatures and throttle reasons; an AMD card's PCIe link is the narrowest link up to the root port, as
  `--bench` reports it (an R9700 on OCuLink: Gen4 x4, not the card's internal x16) - AMD has no PCIe throughput
  counter on RDNA 3/4, so `pcie_rx_mb` stays empty; each model and `STRATA_GLM_MIRROR` drive's MB/s
  (`hardware.drives`); swap, page cache and memory pressure (Linux PSI); the CPU per core (load, clocks, clusters),
  its temperature, power and instruction sets (`cpu_*`); drafts accepted / offered per request and in `totals`. A
  value that cannot be read is `null` (shown as `--`), never 0. CPU power comes from RAPL, which most Linux
  distributions let only root read: `sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj` (until the next
  boot) shows it; an AMD APU falls back to its own power table.

### Abyss Terminal (`./abyss.sh --terminal`)

A terminal UI for the server - this PC's or another's - that you open and close without stopping the model.

```sh
./abyss.sh --terminal                       # the running server, else the last-used config's (Enter starts it)
./abyss.sh --terminal --config abyss-dsv41-iq2xxs-r9700l.json
ABYSS_API_KEY=... ./abyss.sh --terminal --url http://192.168.1.20:8082 --key-env ABYSS_API_KEY   # another PC
./abyss.sh --terminal --plain               # no Textual: plain text redrawn every second (an old Windows console)
./abyss.sh --terminal --once                # one reading printed; --json the same as JSON (issue reports)
./abyss.sh --terminal --demo r9700          # recorded readings, no server (r9700, lowspec, split, multi4, multi8,
                                            #   multi10, loading, starting, down)
./abyss.sh --terminal --layout devices      # the Overview as one panel a device (default for 3+ cards)
```

Windows: `START-ABYSS.bat --terminal` with the same options; without abyss.sh: `python tools/abyss_terminal.py ...`.
The screen is Textual, installed into `.venv` from `third_party/wheels` the first time (nothing is downloaded).

- **Views** (`1`-`8`, `←` `→`): **Overview** - decode and prompt tok/s in big (live while it works, else the last
  request's, labelled), where a token's time goes as one bar (SSD wait / GPU, the CPU or iGPU lane alongside - it runs
  in parallel and is never added) with `--bench`'s verdict and advice (the same code, `tools/bench_rules.py`), the
  busiest card, VRAM, the hottest card (junction too), power, the PCIe link while working, CPU, disk (each SSD of a
  `STRATA_GLM_MIRROR` pair), context fill, experts in VRAM, the RAM tier, system RAM with swap and memory pressure, the
  drafter's acceptance, every card (main / split / helper / iGPU lane) and the last requests; **Speed** - the
  breakdown with the verdict's evidence, SSD reads and waits per drive and the mirror's share, the lane, the drafter,
  the last `--bench` against now; **Memory** - where the experts live, KV, RAM, and what is slowing you down;
  **Requests** - all kept requests (`↑` `↓`: the engine's per-token line of one answered while the Terminal was open);
  **Hardware** - per card clocks, edge / junction / memory temperatures, throttle reasons, PCIe, the CPU per core and
  cluster, the drives; **Logs** - the engine log coloured (`/` filters); **Connect** - the API addresses and a check
  (`h`); **Models** - the installed configs: `Enter` starts one (the running server stops first), `s` stops, `B` runs
  `--bench` (its output in Logs), `y` copies the bench line.
- **Two Overview layouts** (`l` switches, `--layout speed|devices`; the choice is remembered in
  `~/.config/abyss/terminal.json`, on Windows `%APPDATA%\abyss\terminal.json`): **Speed** - the big numbers, the
  token's time and the tiles above; **Devices** - a small speed strip, then every GPU (main / split / helper / iGPU
  lane), the CPU and every drive in a panel of its own, in a grid that wraps (10 cards fit). Until you choose, a config
  with 3+ cards opens on Devices. The footer names the layout.
- **Several GPUs** (2-10 cards, mixed generations): each card shows its role, load, VRAM, the layers and expert pool
  it holds in a layer split (`L0-13`, from the engine log on this PC), temperatures, power against its own limit, its
  PCIe link against its own max, and its share of the token time (from the engine's STAT line, v1.3.80). The busiest card is starred;
  a card whose link fell while working, or that sits on a link under half the others' (an x1 riser, Gen2/Gen3 x4), is
  pink and named in a banner. Every drive shows its MB/s and its share of the expert reads (any number of drives). At
  80x24 the cards are summed up in one line (busiest / VRAM / hottest / lowest link) with the cards that matter.
- **Detached:** a server the Terminal starts runs in a session of its own (`abyss.py --config <config> --plain` with
  `ABYSS_NO_BROWSER=1`, so no browser opens; output in `<config>.server.log`, PID in `<config>.pid`) - quitting asks
  *keep serving* or *stop it*; a server that
  was already running is left alone. A stop is `POST /api/shutdown`, else Ctrl+C to the server's process group; after
  60 s it is killed. The next `--terminal` attaches again.
- **The header** always names the model with its quant (e.g. *DeepSeek V4.1 Flash 1017 UD-IQ2_XXS*) and shows the
  settings that change the answers as badges (`UNCENSOR`, `TAIL SKIP n`, `DEFER`, `ABL`; `MTP`, `MIRROR`, `HELPER`).
  States: connecting, loading (the warm-up's progress, from the engine log), idle, reading the prompt, generating,
  queued, reloading, error, stale (the last values kept), disconnected (it retries with back-off), auth required,
  not Abyss, stopped.
- **Live per-token time (a 1.3.80+ engine):** the engine's STAT line (every 8 tokens) carries the SSD wait, the
  lane's experts and busy ms, misses, the tail skip, the mirror's MB per drive, the drafter's accepted / offered /
  verify steps, and a window over the last ~64 tokens. While an answer is written the Token time tile and the Speed
  view show that window (`live · last ~64 tokens`) and `--bench`'s verdict follows it; idle, the last answer's final
  numbers (`last answer · engine STAT`); with an older engine the engine log after each answer (`engine log`). The
  Speed view adds the lane's estimated bandwidth (experts a token x one expert's MiB / busy ms), the mirror's split as
  the engine counts it, the drafter's live acceptance and tokens a verify step; the prompt tile shows the reused tokens
  and the SSD wait so far while a prompt is read. The header's badges (`TAIL SKIP`, `DEFER`, `MIRROR`, `HELPER`, the
  drafter) come from the engine's INFO line, so a remote server shows them too.
- **A server still loading (1.3.80+):** the port is bound before the engine starts, so `/health` answers
  `status: "loading"` with the stage, the warm-up's percent and GB and the seconds so far - the Terminal shows it on
  another PC as on this one. From v1.3.81 that answer is a 503 with `Retry-After: 5` (a client that reads 200 as
  "ready", such as a llama-swap health check, waits for the engine), and 200 once the model is ready; every other
  request gets 503 + Retry-After until then. A context reload
  shows the same under `reloading`.
- **`--` means not reported**, never 0. Still waiting for an engine counter: the helper card's share of the token
  time (E6, needs two CUDA cards).
- **Remote:** read-only (no Models, which reads this PC's configs). The key is read from the environment variable
  `--key-env` names and never stored; over plain `http://` to another PC it travels readable - use Tailscale or an SSH
  tunnel (`ssh -L 8082:127.0.0.1:8082 host`, then `--url http://127.0.0.1:8082`). A 1.3.80+ server sends its engine
  log's tail over `GET /api/log?since=<offset>&max=<bytes>` (the key is needed; the API key's values are masked), so
  Logs and the per-request breakdown work remotely too; an older server leaves Logs with a hint.
- **Keys:** `l` Overview layout, `r` poll now, `space` freeze, `t` theme (dark / light / mono; `NO_COLOR` starts in mono), `L` records a
  temperature CSV to `bench/thermal/` every 2 s, `b` opens the web app, `c` copies the API address (OSC 52), `?` help,
  `q` / Ctrl+C quit. It fits 80x24 (compact boxes, the view scrolls) and shows more from 110 columns.

### With Strata-tools' strata.sh

If you use the `strata.sh` helper (start / stop / log), put a run script and the config into the Abyss folder under
its naming (`run-<name>.sh`, `strata-<name>.json`) and point it there: `STRATA_DIR=~/llm/apps/Abyss ./strata.sh start <name>`.
The run script is the start command above (`exec .venv/bin/python serve/server.py --engine strata "--config" "<config>" "--port" "8082" "$@"`).

### Coding tools: Claude Code and OpenCode (`--connect`)

`./abyss.sh --connect <tool>` asks the server that runs now (`GET /v1/models`: its model name and context) and its
run config (address, API key) and prints the settings for that tool. Nothing is written unless you add `--write`.

```sh
./abyss.sh --connect claude              # the variables to start Claude Code with
./abyss.sh --connect claude --write      # writes run-claude-<config>.sh next to the configs (asked first)
./abyss.sh --connect claude --check      # one prompt + one tool call the way Claude Code sends them
./abyss.sh --connect opencode            # the provider block for opencode.json
./abyss.sh --connect opencode --write    # merges provider "abyss" into ~/.config/opencode/opencode.json (backup kept)
./abyss.sh --connect opencode --check    # the same check in OpenAI form
ABYSS_KEY=... ./abyss.sh --connect claude --url http://192.168.1.20:8082 --key-env ABYSS_KEY   # another PC
```

- **Claude Code:** the launcher sets `ANTHROPIC_BASE_URL`, every model slot (`ANTHROPIC_MODEL`, the Opus / Sonnet /
  Haiku defaults, `CLAUDE_CODE_SUBAGENT_MODEL`) to the server's model, `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the
  server's context (Claude Code assumes 200K for a model it does not know and would compact too late),
  `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (a quarter of the context, at most 32K), `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`
  and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, for that run only - `~/.claude/settings.json` is not changed. The
  key goes as `ANTHROPIC_AUTH_TOKEN` (Bearer) from `$ABYSS_API_KEY`; `ANTHROPIC_API_KEY` is unset, as Claude Code would
  ask whether to use it. If your `~/.claude/settings.json` has `ANTHROPIC_*` in its `env`, those win - take them out.
  Claude Code's own prompt is about 20K tokens: set the server up with a context of 32K or more (64K+ is comfortable).
  A model name that starts with `claude-` is taken for one of Claude's models: give the config another `model_name`.
- **OpenCode:** provider `abyss` (`@ai-sdk/openai-compatible`, base URL `.../v1`, `limit.context` = the server's
  context, the key as `{env:ABYSS_API_KEY}`). Your default model is kept if you have one; pick `abyss/<model>` with
  `/models`. A config with comments (JSONC) is not touched: copy the printed block by hand.
- **Temperature 0 for coding agents (optional).** The config keeps the model card's sampling (e.g. Qwen3.8:
  temperature 1.0, top_p 0.95, top_k 20). For coding and agent work you can set `"sampling": {"temperature": 0.0,
  "top_p": 1.0, "top_k": 1}` in `abyss-<name>.json`: the model always takes its most probable token, so the drafter's
  guesses match more often - on an RTX 4090 with Qwen3.8-Flash-Next the drafts accepted went 76 -> 86% and a long coding
  prompt ran ~20% faster ([#39](https://github.com/zephel01/Abyss/issues/39)). Greedy answers can repeat themselves on
  open-ended text, so keep the default for chat and writing. A request's own `temperature` wins over the config.
- The key is never written into a file. Tested with Claude Code 2.1.316 against Qwen3.8-35B-A3B-Distill on one RTX 5090
  (a file-writing tool call).

## Several GPUs

With two or more NVIDIA GPUs the layers can be split across them (a layer split). Each GPU keeps only its own layers'
experts in VRAM, so the second card's VRAM holds more of them.

- **Setup:** `./setup-deepseek.sh --gpus 0,1` or `--gpus all`. The config gets `"gpu": [0, 1]`, and the server
  starts the engine with `--layer-split auto`. GPUs of different generations (an RTX 5090 with an RTX 3090) are
  built for both architectures (`86;120a`).
- **What `auto` picks (v1.3.25):** it prices every boundary - and each card by itself - by the decode time it
  predicts (each card's bytes over its memory bandwidth, the misses the CPU / iGPU lane does not hide, a fixed cost
  a boundary; fitted on an RTX 5090 + RTX 3090, within 0.35 ms of the measured 10.9 - 18.1 ms), and runs on one GPU
  when that is faster: a decode token goes through the parts one after another, so a slower second card's layers can
  cost more than its VRAM saves. RTX 5090 + RTX 3090 (both PCIe Gen4 x4): the 5090 alone, 94-95 tok/s (the split it
  picked before, 26 / 17: 84 tok/s; the best split, 33 / 10: 88 tok/s). `STRATA_GLM_SPLIT_KEEP=1` splits anyway
  (more context or VRAM for prompts), `STRATA_GLM_SPLIT_LOG=1` prints every prediction.
- **Choosing the boundary:** `"layer_split": "22"` in the config (the second card's first layer).
- **The iGPU lane:** it works with a split too. One iGPU computes every GPU's RAM tier, one call at a time.
- **What has been checked:** a two-part split on one GPU (`STRATA_GLM_SPLIT_SAME=1 STRATA_GLM_SPLIT=22
  STRATA_GLM_DEVS=0,0`, for testing) matches the unsplit model (KL within the run-to-run spread), and an RTX 5090
  (PCIe Gen4 x4) + RTX 3090 (a USB4 enclosure, ~3.8 GB/s) split 27 / 16 ran at 8K / 32K prompts of 1,506 / 1,985
  tok/s and decode of 54.9 / 56.7 tok/s with the iGPU lane (one RTX 5090 + iGPU: 1,683 / 2,876 and 58.5 / 65.9).
  The second card raised the VRAM hit rate (82 -> 91%) but the hop between the cards and the slow link cost more.
  A fast second card on a real PCIe slot has not been measured.
- **Speculative decoding** (v1.3.26): the verify window runs part after part, each part's columns crossing to the
  next as a token's residual does. RTX 5090 + RTX 3090 split 33 / 10: 87-90 tok/s (without it ~80-88), the drafts
  accepted as often as on one card (57% on code).
- **Prestaging** (v1.3.27): each part copies its next layer's predicted RAM-tier experts during the mixer, as one
  GPU does; on a split the chunk is sized first (8192 at most) and the prestage buffer takes what it leaves. RTX 5090 +
  RTX 3090 split 33 / 10: 32K-token prompts 2,177-2,186 -> 2,274-2,350 tok/s, 8K about even. `STRATA_GLM_PRESTAGE=0`
  turns it off.
- **Qwen3.8-Flash-Next (qwen4exp, BETA from v1.3.109):** the split runs its token path, prompts and verify windows; the MTP drafter
  (`STRATA_Q4X_MTP`) sits on the last part (it reads the final residual and the head; that part also keeps a pinned
  copy of token_embd, +497 MB of RAM). The first boundary is layer 2 or later (the first part applies the per-layer
  embeddings). `auto` counts its delta-net states, K / V caches and QSA keys at the context, and the drafter on the last
  part. Checked on one RTX 5090 seen as two devices (M16b build, split 24): KL the same as unsplit, real-use tasks the
  same, drafts accepted 84.7% (83.2% unsplit). On a real 16 GB + 16 GB GeForce pair (#35) a split at layer 32 was 22%
  faster on long requests, but `auto` still picks the first card alone there (its cost model is to be refitted with
  those numbers): set `--layer-split 32` by hand. From v1.3.110 a hand split is checked against the model's own layer
  count (before, `--layer-split 46` on the 48-layer model was refused), and when `auto` runs on one card, that card
  keeps its own `STRATA_GLM_VRAM_GB_PARTS` entry. Reports from two-card machines are welcome (the bench issue form).
  Conversation reuse (snapshots) is off under a split.
- **How the rows cross a boundary** (testing): `STRATA_GLM_HOP=host` (default: the pinned host buffer), `peer`
  (device to device; cards without peer access use host) or `sock` (a byte stream, the shape a link to another process
  or PC will take). On one RTX 5090 seen as two devices all three give the same KL and 44.6-46.1 tok/s.
- **A memory cap per part** (testing): `STRATA_GLM_VRAM_GB_PARTS=14,14` gives each part its own
  `STRATA_GLM_VRAM_GB`, in part order; `auto` sizes the parts with the same caps.
- AMD GPUs: see "AMD GPUs" below.
- **V100 (sm_70):** CUDA 13 cannot compile for the V100, so install CUDA 12 (12.9) and build with
  `./setup-deepseek.sh --nvcc /usr/local/cuda-12.9/bin/nvcc` (CUDA 12.9 also builds an RTX 50's `120a`). The sm_70
  build has been checked to compile; it has not run on a V100 yet. The newer tensor-core kernels (the indexer and the
  prompt attention) need sm_80 or newer, so a V100 uses the older ones.

## AMD GPUs (HIP, experimental)

DeepSeek V4 also runs on AMD Radeon cards: the RX 7900 XT / XTX (gfx1100), the RX 9070 / AI PRO R9700 (gfx1201) and
the Ryzen AI Max's Radeon 8060S (gfx1151).

- **Setup:** `./setup-deepseek.sh --backend hip` (chosen by itself when there is no NVIDIA GPU). On Linux it uses ROCm 10
  in `/opt/rocm` (or `/opt/rocm/core-10.0`). It needs HIP 7.14 or newer and stops on ROCm 7.0-7.2, which fail with
  current drivers (issues #1 and #8; `ABYSS_OLD_ROCM=1` builds with one anyway). The engine is built in
  `build-hip/strata`, and the config gets `"backend": "hip"`. After switching to another ROCm, run it again (with
  `ROCM_PATH`): it builds the engine from scratch for that ROCm (v1.3.44). On Windows with Visual Studio's MSVC 14.50
  or newer, setup moves one include line in the ROCm SDK's `__clang_hip_runtime_wrapper.h` (LLVM's fix for
  "'isgreater' cannot overload", issue #43; the original is kept as `.abyss-orig`); `ABYSS_NO_ROCM_HEADER_FIX=1`
  leaves the SDK as it is.
- **Driver and ROCm:** [docs/amd/README.md](amd/README.md) has the install steps (amdgpu, ROCm 10's per-architecture
  packages - an R9700 needs `amdrocm10.0-gfx1201` - and why `HSA_OVERRIDE_GFX_VERSION` must not be set), a check
  list, and troubleshooting.
- **What has been checked:** the Radeon 8060S (gfx1151) alone, and an AI PRO R9700 (gfx1201) with the 8060S as its
  iGPU lane: DeepSeek V4 over swe-bench tasks 36-42 tok/s in v1.3.39 and 51.7 in v1.3.41 (49.6 without the lane), one
  token at a time 52 tok/s in v1.3.40 (46 without the lane in v1.3.41), KL against the CUDA build within the
  run-to-run spread (0.005-0.006). With the DSpark drafter (v1.3.43) the same swe-bench tasks run at 61.9 tok/s
  without the lane and 67.3 with it ([The DSpark drafter](#the-dspark-drafter-deepseek-v4-flash)).
- **ROCm's hardware queues:** keep `GPU_MAX_HW_QUEUES` at 2 (setup writes it); 1, which R9700 guides suggest, cuts
  the prompts by a third ([docs/amd](amd/README.md#3-environment-variables)).
- **A discrete Radeon on a Ryzen AI Max:** the iGPU computes the RAM-tier experts as it does beside an NVIDIA card
  (+12-46% decode on the R9700); the config's `"gpu_pci"` keeps the right card when HIP numbers the GPUs otherwise.
- **What is faster:** the indexer's scores and the prompt attention use RDNA's WMMA (gfx11 / gfx12).
  `STRATA_DS4_IDX=1` keeps the older indexer kernels, and `STRATA_DS4_ATTN=1` the older attention.
- **Limits:** mixing it with NVIDIA in one process (CUDA and HIP together) is not supported.

### Measuring this PC (`./abyss.sh --bench`)

`./abyss.sh --bench` (Abyss stopped, a few minutes) measures the installed model the same way on every PC: decode on
three questions, prefill at 2k / 8k tokens, then where a token's time goes (SSD waits, the CPU / iGPU lane, PCIe
copies, the GPU), the likely bottleneck with what moves it, and a quality check against a known-good run of the same
model (`bench/ref`). `--deep` adds the GPU timeline and a sweep of the PCIe / lane share; `--full` repeats the answers
and adds a 32k prompt. It writes `abyss-bench.txt` to paste into an issue. [BENCH.md](BENCH.md) explains each line.

### Tuning the lane for this PC (`./abyss.sh --calibrate`)

Run it once after the setup (Abyss stopped, ~10-15 minutes, the model loads first): it measures decode with a few
splits of the RAM-tier experts between the CPU lane and the PCIe link, and with fewer CPU threads, and keeps a setting
that is more than 3% faster than the engine's own choice (`STRATA_GLM_PCIE_SHARE`, `STRATA_GLM_CPU_LANE` in the
config's `env`, and `~/.config/project-maya/calibration.json` so a setup again keeps it). The best split depends on
the PC: on an RTX 4090 + 96 GB with Qwen3.8-Flash-Next UD-Q4_K_XL the share went 0.08 -> 0.10 and decode 50.2 -> 56.5
tok/s (+12%, [#39](https://github.com/zephel01/Abyss/issues/39)). Run it again after a new GPU, more RAM, another
model or a setup that rewrote the config.

### Checking that the model runs right (`./abyss.sh --check`)

`./abyss.sh --check` (v1.3.112, Abyss stopped, 15-30 minutes; `--quick` about 5, `--long` adds the biggest prompts the
context allows) runs the installed model the way it is used and checks that it stays up and answers right: the answer
with and without the drafter, Japanese, small coding tasks whose code runs against tests, a tool call, prompts of 8K /
24K / 32K tokens up to the context's top with a needle, and an 11-turn conversation. It writes `abyss-check.txt` (no
host, user or folder names; nothing is sent) with this PC, the model and "What limits this PC" - the estimate of which
part (GPU, CPU lane, PCIe, RAM, SSD, VRAM) holds a token back. Attach it to a [bug report](https://github.com/zephel01/Abyss/issues/new?template=bug.md)
(`--report` includes it). The exit code is 1 only when a step fails. Before the setup, or with `--setup` / `--gpu` /
`--gpus`, `--check` checks the PC only, as before. Windows: `START-ABYSS.bat --check`. Each step:
[BENCH.md](BENCH.md#the-stability-check---check).

### Japanese finance rules (`./abyss.sh --check --finance`)

`./abyss.sh --check --finance` (v1.3.123, Abyss stopped, about 10-30 minutes) asks the installed model 17 Japanese
questions on synthetic accounts: FX / CFD / crypto margin, 信用取引 margin and margin calls, swap points and dividend
adjustments kept apart from price P/L, listed-stock tax, NISA limits, a foreign-currency deposit's yield; three where a
needed figure (a collateral haircut, a loss-cut rate, a swap rate) is not given and the right answer says so; and one
that asks for a firm buy / sell call, which it should not give. Each question **states the rule it uses** (the rate,
the floor, the tax rates), so a model is not marked down for not knowing a rule that changed after its training; the
rules live in one table in `tools/finance_check.py` with their sources and dates (`RULES_VERSION`). Only what does not
depend on the market is graded, from the final `答え:` lines (not the thinking). It writes every answer and grade to
`abyss-finance.txt`; the score is "n of 17 items as the rules say (rules <version>)" - not investment advice, and not
a measure of how good a model is at finance. Any OpenAI-compatible server: `python tools/finance_check.py --url
http://127.0.0.1:8080`.

### Tuning for your GPU (`./abyss.sh --gpu-tune`)

Some kernel choices depend on the GPU; the engine picks them per GPU architecture (it logs `glm fast: dense GEMV
tune`), from measurements on the GPUs we have (an AI PRO R9700, an RTX 5090). `./abyss.sh --gpu-tune` (v1.3.51)
measures them on your GPU(s) without a model, in under a minute each, with the model server stopped:

- what the GPU is (architecture, compute units, VRAM) and a memory copy's bandwidth;
- the Q6_K dense GEMV, 4 warps a row or one, over the models' row counts (an R9700: one warp, 560 -> 510 us).

A choice clearly faster than the engine's own (2% or more) goes into the installed configs' `env`
(`STRATA_GLM_MV_Q6K_SPLIT`) and applies at the next start. `abyss-gpu-tune.txt` gets a report: for a GPU other than
those two, please paste it into an [issue](https://github.com/zephel01/Abyss/issues/new?template=gpu-tune.md) - the
engine's defaults for that GPU come from such reports. From v1.3.115 it also finds an engine built by hand in another
folder (the configs' `exe`, e.g. `build-cuda`) instead of stopping with "not built here yet" (#37).

## NVIDIA DGX Spark (GB10, ARM, beta)

From v1.3.113 the engine builds for aarch64 + sm_121 and runs on a DGX Spark's GB10, whose GPU and CPU share one
memory: the expert pool and the RAM tier are sized from one budget (`STRATA_GLM_UMA_HEADROOM_GB` left free),
programmatic dependent launch is on (`STRATA_GLM_PDL`), RAM-tier experts are copied into the pool's slots
(`STRATA_GLM_UMA_DIRECT=1`: read in place) and the CPU lane is off. Measured on a rented Spark with Qwen3.8 16G64G:
47.05 tok/s with the MTP drafter (CLI, 4K), 43-54 through the server, 8K prompts ~1,040 tok/s. With v1.3.114
(2026-11-06): GLM-5.3-Flash 19-20 tok/s, DeepSeek V4 0731 21.0 (22.8 with the DSpark drafter), DeepSeek V4.1 Compact
~18; GLM's long prompts through the server are still slow there (~120-130 tok/s, the CLI 675-796; open). From
v1.3.115 the engine does not start on memory another process holds (`STRATA_GLM_UMA_SHARE`,
`STRATA_GLM_UMA_OTHERS_GB`). Build steps, the measurements and what is untested: [docs/DGX-SPARK.md](DGX-SPARK.md).

## DeepSeek V4.1 Flash

From v1.3.29 the same engine runs DeepSeek-V4.1-Flash-1017. `./setup-deepseek.sh --model v41` (v1.3.100) installs
zephel01's LDLQ GGUF ([Setup in one command](#setup-in-one-command)); other GGUFs are set up by hand.
[docs/deepseek41/README.md](deepseek41/README.md) has the GGUFs that load, the pack command, a config and the
measurements, and [docs/deepseek41/SPEC.md](deepseek41/SPEC.md) what the model changes against V4. In short:

- **Disk and memory.** nazeshinjite's Q2 GGUF is 366 GB: 153 GB of experts (384 a layer, 9.95 MB each), ~200 GB of
  Engram tables and the dense weights. On an RTX 5090 + 94 GB RAM ~57% of the experts are resident (VRAM ~2,300, a
  60 GB RAM tier ~6,500); the rest and every Engram row (48 a token, 264 bytes each) are read from the GGUF. Keep it
  on the fastest NVMe you have, and leave ~25 GB of RAM to the page cache (`STRATA_GLM_RAM_GB=60` on a 94 GB box).
- **Speed is the SSD's.** A token that reads no expert from the SSD takes 13-15 ms; each read adds ~3-4 ms at
  3.4 GB/s. With `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1` (a route's 4th-6th experts left out when they
  are only on disk; KL 0.229 -> 0.240 against the unquantized model) decode is ~28 tok/s, 17.7 without; swe-bench
  coding tasks ran at 51.6 tok/s end to end. Prompts are slower than V4's: the prompt's experts come from the SSD (the
  attention runs batched since v1.3.60).
- **A lighter dense pack** (v1.3.36, optional): `attn_q_b` / `attn_output_a` / `attn_output_b` at Q5_K instead of
  Q8_0 (dense 8.14 -> 6.50 GB, 177 more VRAM expert slots) decodes 10-15% faster for KL 0.242 -> 0.251; see
  [docs/deepseek41/README.md](deepseek41/README.md#a-lighter-dense-pack-q5_k).
- **Start it in a memory-capped scope** (`systemd-run --user --scope -p MemoryMax=88G ...`): the page cache and the
  pinned tier together can push systemd-oomd into killing the desktop session.
- **Speculative decoding** (v1.3.53, opt-in): Lucebox's DSpark drafter for V4.1, set by hand with
  `STRATA_DS4_DSPARK=<file>` (the setup's `--dspark` installs V4 0731's drafter, not this one; Lucebox's V4.1 drafter
  repository is no longer on Hugging Face, v1.3.100), with the n-gram window; the model's own MTP layers are in no GGUF.
- **Images** (v1.3.93, PR #28; ISTA-DASLab's new-format templates from v1.3.105): with ISTA-DASLab's V4.1 mmproj and
  the image router bias, set up by hand: [docs/SETTINGS.md](SETTINGS.md#images-on-deepseek-v41).
- **Not yet:** a layer split, conversation slots on disk.

## Qwen3.8-Flash-Next

From v1.3.74 the engine runs Qwen3.8-Flash-Next (llama.cpp's `qwen4exp`) on the fast path (`--glm-pack`), with its
prompt path, verify window and MTP drafter. It is set up by hand; [docs/qwen38/README.md](qwen38/README.md) has the
GGUFs that load, the pack command (`tools/iq_pack.py --fast`), the drafter, a config and the measurements. In short:

- **Our builds:** [16G64G](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF) (KL 0.083; RTX 5090 at
  32 GB 128.7 tok/s with the MTP drafter, at 16 GB 95.1) and
  [12G32G](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF) (L448: 448 of 512 experts, 67.5 GB, for a
  12 GB GPU + 32 GB RAM).
- **Other GGUFs** (beta): NVIDIA's NVFP4 checkpoints (v1.3.101) and unsloth's UD 3.1 sets (v1.3.112-v1.3.115), each
  with its numbers in docs/qwen38.
- **Two GPUs** (beta, v1.3.109): see [Several GPUs](#several-gpus). **DGX Spark** (beta, v1.3.113): see
  [NVIDIA DGX Spark](#nvidia-dgx-spark-gb10-arm-beta).
- **The DFlash 2 drafter** (opt-in, v1.3.125): `STRATA_Q4X_DFLASH=<the GGUF>` loads nazeshinjite's Q8_0 GGUF of
  incoai's [DFlash 2](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF) block drafter (0.65 GB file,
  0.76 GB of VRAM) instead of the MTP layer: five small layers read the model's residual entering layers 6/16/26/36/46
  and propose up to 5 tokens in one pass; the verify window checks them as it checks the MTP drafts.
  `./setup-deepseek.sh --model qwen38 --dflash` downloads it and names it in the config (`--no-dflash`: off). It is
  faster than the MTP layer on code and lists and slower on prose (Japanese most), so it is not the default:

  | 16G64G, RTX 5090 (32 GB), greedy 512 tokens, tok/s | code | list | Japanese prose | swe-style fix |
  | --- | --- | --- | --- | --- |
  | no drafter (n-gram only) | 91.0 | 93.6 | 93.6 | 99.9 |
  | MTP layer (`STRATA_Q4X_MTP`) | 140.2 | 137.2 | **105.7** | 128.2 |
  | DFlash 2 (`STRATA_Q4X_DFLASH`) | **152.3** | **142.1** | 91.9 | 122.0 |

  On 12G32G (L448v2; the card capped at 12 GB, a 20 GB RAM tier in a 28 GB cgroup, 8 CPU threads) the two drafters
  are even: code 58.0 / list 72.6 / Japanese 46.8 / swe 37.9 tok/s with DFlash 2 against 58.5 / 72.6 / 47.9 / 36.7
  with the MTP layer (no drafter: 48.1 / 63.7 / 47.0 / 41.1).
  One GPU only (not with a layer split). `STRATA_Q4X_DFLASH_N` (5, 1-5): drafts a step at most;
  `STRATA_Q4X_DFLASH_PMIN` (0.9): the selector's confidence a draft needs; the walk stops at the first draft below it
  (0.75, the drafter's own default: list 137 against 142; 0.5 slower everywhere). The drafts match a numpy reference
  of the drafter (teacher-forced code: 3.62 of 3.83 drafts kept a step against the reference's 3.59 of 3.79).
  A draft step runs as one CUDA graph (1.14 ms on an RTX 5090, was 1.96); `STRATA_Q4X_DFLASH_GRAPH=0` launches its
  kernels one by one (the same drafts).
  On a PCIe-bound card the two are a tie: an RTX 4090 + 96 GB with the full UD-Q4_K_XL (greedy, a long code generation,
  8K tokens) ran 55.6 tok/s with the MTP layer and 56.4 with DFlash 2, although DFlash 2 kept 86-89% of its drafts
  against MTP's 78% (#39). There decode waits on the experts fetched over PCIe (`vram hit` ~60%, ~20 RAM fetches a
  token): a better drafter yields more tokens a verify window, but each window still pays the same expert fetches.

## Ornith-1.5-122B

From v1.3.46 the engine also runs [Ornith-1.5-122B](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF), a
Qwen3.5-MoE model (llama.cpp's `qwen35moe`): 36 gated delta-net layers and 12 full-attention layers, 256 experts a
layer with 8 routed and a gated shared expert, a plain residual. `./setup-deepseek.sh --model ornith` (v1.3.100) installs zephel01's IQ3_XXS-B for a 16 GB GPU +
64 GB RAM; other GGUFs are set up by hand. [docs/ornith/README.md](ornith/README.md) has the GGUF, the pack command, a config and the measurements. In short:

- **Memory.** unsloth's UD-Q4_K_XL is 79.7 GB: 70.4 GB of experts (256 x 48, 3072 x 1024, IQ4_XS-Q6_K) and 5.4 GB of
  Q8_0 dense weights that stay on the GPU. A 32 GB card holds ~25 GB of experts; a 60 GB RAM tier the rest, so on a
  94 GB box decode reads nothing from the SSD.
- **Speed.** AI PRO R9700 + 8060S lane: decode 23-25 ms a token (41-43 tok/s; the CPU lane instead of the iGPU:
  30 ms), a 575-token prompt at 359 tok/s through the batched path. KL 0.024 against llama.cpp over 559 positions - llama.cpp's own CPU and HIP backends differ by
  0.027-0.029 on this model.
- **Speculative decoding.** The n-gram drafts run through the verify window (one draft a window by default,
  `STRATA_DS4_SPEC=<k>` more): a copy task 57 tok/s with one, 76 with three. The GGUF's NextN block drafts too once
  its experts are repacked (`tools/nextn_pack.py`, `STRATA_GLM_MTP_GGUF`): code 59 tok/s, prose 50, the copy task 85
  (without the iGPU: code 36 -> 42, prose 33 -> 39).
- **NVIDIA:** on an RTX 5090, IQ3_XXS-B decodes at 116 tok/s with its NextN drafter (75% accepted, v1.3.100); with a
  1.45 GB expert pool (issue #23's 8 GB card) prompts ran at 356 tok/s (2K) and decode at 45.5 (v1.3.82).
- **Small VRAM:** `--kv q4_0` (or `k8v4`, `int8`) shrinks the attention cache to 28% (41%, 53%) of fp16 and gives the
  rest to the expert pool (v1.3.134, #44): Ornith-1.5-397B at a 128K context starts on a 16 GB card. See `--kv` in Settings.
- **Not yet:** conversation slots, a layer split, images; the expert kernels are not yet tuned for its shapes.

## EXL3 quants (GLM-5.3-Flash)

From v1.3.35 the engine also reads [exllamav3](https://github.com/turboderp-org/exllamav3)'s EXL3 quants of
GLM-5.3-Flash, in place from their safetensors; [docs/exl3/README.md](exl3/README.md) has the details. In short:

- **Installing.** `./abyss.sh --setup` lists turboderp's `EXL3-sc_2.10bpw` (88.3 GB) and `EXL3-3.05bpw` (125.3 GB)
  beside Maya's GGUFs and says what each suits on this PC: recommended when RTX 30+ cards hold 80% of it, a quality
  option with an APU iGPU or an AVX-512 CPU for the RAM tier, not for a CPU with AVX2 only. `--model EXL3-sc_2.10bpw`
  skips the question. NVIDIA GPUs only in the setup.
- **Switching.** Each quant gets its own config and start script (`maya-exl3-sc_2.10bpw.json`,
  `run-maya-exl3-sc_2.10bpw.sh`); a GGUF install and an EXL3 install sit side by side. The engine knows an EXL3 pack by
  its `pack/exl3.txt` - no setting.
- **By hand.** `.venv/bin/python tools/exl3_pack.py --model <dir>` (~5 s) writes `<dir>/maya-exl3.gguf` and
  `<dir>/pack/`; start with `--glm-pack <dir>/pack`.
- **What to expect** (RTX 5090 + Radeon 8060S lane + 94 GB RAM, 1,000 scored tokens): sc_2.10bpw nll 1.591-1.598 and
  top-1 63.3-63.7% against Maya-S-v2-IQ2_XXS's 1.634 / 62.3%, at 18.8 ms a token against 15.4 (v1.3.36: the iGPU's
  trellis decode by `v_sad_u8` and `v_dot2_f32_f16`; 22.4 in v1.3.35).

## Memory and speed

The engine fills the GPU's free VRAM with experts (after the dense weights, 6.2 GB for UD-Q2_K_XL, the caches and the
prompt buffers), sizes a pinned RAM tier from the free RAM, and reads the rest from the SSD when needed. It records
which experts you use (`<pack>/expert_usage.txt`) and loads those first next time.

Measured on one RTX 5090 + Ryzen AI Max+ 395 with 94 GB RAM (UD-Q2_K_XL, context 65536, every expert in VRAM 23 GB +
RAM 70 GB; the iGPU lane on):

| | |
| --- | --- |
| Decode, 1,024-token answers (prose / code, greedy) | ~92-95 tok/s (CPU lane instead of the iGPU: ~55-60) |
| Decode, 16 swe-bench coding tasks x 3 (sampled, end to end) | ~100 tok/s (10.1 ms a token averaged over the run; 1,024-token answers: a single token 11.1 ms, the 2-token verify window 14.6) |
| Decode, 60 swe-bench coding tasks x 5 (sampled, end to end) | ~107 tok/s (100.5-114.5 a task; v1.3.27) |
| Decode at a 125K-token context | ~65 tok/s |
| Prompt, 8K / 32K / 125K tokens (server, second run) | ~1,630-1,760 / ~3,100-3,270 / ~2,800 tok/s |
| Start-up (weights + expert warm-up) | ~1 min |
| Attention caches at 65536 tokens | 0.44 GB |

(Decode rows: Abyss v1.3.21 with the Q5_K attention pack ("A lighter dense pack", above; the GGUF as published:
~82-84 and ~91-94). Prompt and 125K rows: v1.3.16-1.3.18, the GGUF as published. On the same machine and file
llama.cpp `bed0a8566` with `-ngl 99 -ncmoe 32` decodes at 28-30 tok/s
and reads prompts at 78 tok/s with ubatch 512, 660 with ubatch 8192.) The numbers of every release, and how each was
measured, are in [CHANGELOG.md](../CHANGELOG.md).

The first prompts after a start, or with less memory than the model, read experts from the SSD and are slower. The
CPU computes part of the RAM-tier experts while the GPU copies others over PCIe; the split is measured at start.

**The iGPU lane.** On an AMD APU (Ryzen AI Max+ 395: Radeon 8060S) the integrated GPU reads the same RAM as the CPU,
at ~240 GB/s against the CPU threads' ~75. With ROCm installed (the build then makes `libabyss_igpu.so`), the engine
registers its RAM tier with the iGPU at start and lets it compute the RAM-tier experts instead of the CPU threads -
the experts are not copied. The start-up log says which lane it chose and why (`iGPU lane: an expert 0.046 ms vs 0.107
on the CPU's 16 threads`); it keeps the CPU unless the iGPU is clearly faster (1.5x, `STRATA_GLM_IGPU_MARGIN`), with
the exact combine (Qwen3.8), and when the library or a HIP device is missing.
The prompt path uses the lane too: the RAM-tier experts a prompt chunk needs are computed on the iGPU instead of the
CPU threads (an 8K prompt +40%). A lane call has a deadline (`STRATA_GLM_IGPU_TIMEOUT_MS`, 20 ms): when the kernel
driver holds the iGPU's queue (it happens for a few minutes after a start, while the kernel reclaims memory after the
warm-up), the CPU threads take over until the late call finishes, and the start-up timing is repeated until it is
clean.

**Speculative decoding (`STRATA_DS4_SPEC`, on by default).** The GGUF has no draft model, but the answer often repeats
what is in the prompt or earlier in the answer (code, file paths, tool calls). The engine looks up the last 3 tokens'
earlier occurrence, checks the token that followed it together with the next one in one pass, and keeps it when the
model agrees. The output is the same as without it (the same sampler on the same logits); only the time per token
changes. A gate learns each request's acceptance and skips the draft while it would not pay. Coding prompts: +10-13%
(77-85% of the drafts accepted); prose: about even (~50% accepted). `STRATA_DS4_SPEC=0` turns it off; a layer split
across GPUs runs it too (v1.3.26).

## The DSpark drafter (DeepSeek V4 Flash)

From v1.3.43 the engine can run DeepSeek's DSpark block drafter beside DeepSeek V4 Flash 0731 - three DeepSeek V4
layers that read the model's own layer 40-42 outputs and propose the next 5 tokens at once, each with a confidence.
The leading drafts at 0.6 or above go through one verify window together with the token, and each is kept when the
model's own sampling agrees with it, so the answers come from the model as before; only the time per token changes.
The file is bleysg's GGUF of the official drafter,
[`DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf`](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)
(6.94 GB, MIT; Q2_K experts, its Markov head in Q8_0).

**Turning it on.** `./setup-deepseek.sh --dspark` downloads it, checks its SHA-256 and adds it to the config:

```json
"env": {"STRATA_DS4_DSPARK": "/home/you/llm/apps/Abyss-data/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF/DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf"}
```

Then restart the server (`../Abyss-tools/abyss.sh restart`). `./setup-deepseek.sh --no-dspark` (or taking the line
out) turns it off. At start the engine log's `dspark:` line names the file and the VRAM it took (`3 stages over
target layers 40-42, 5 drafts a step; 6.48 GB on CUDA0 ...`). It runs with DeepSeek V4 Flash 0731 on one GPU - not
with a layer split or the MTP block (DeepSeek V4.1 takes its own drafter file, see [DeepSeek V4.1 Flash](#deepseek-v41-flash)); then the line says `dspark: not loaded - <why>` and the model runs
on without it.

**What it costs.** The drafter keeps its weights and all of its experts in VRAM: 6.48 GB that the expert pool gives
up. The RAM tier keeps the size it has without the drafter, so the coldest ~6.5 GB of experts are read from the SSD
instead (0.01 reads a token in the decode on the machine below; `STRATA_GLM_RAM_GB` still sets the tier's size by
hand, for a PC with RAM to spare). Without that cap the tier pinned those experts too: on a 94 GB PC 73.9 GB pinned
left 13 GB free, and an 8K prompt took 301 s instead of 11. A draft takes ~4.4 ms on an AI PRO R9700.

**Speed** (AI PRO R9700, OCuLink PCIe Gen4 x4, in a Ryzen AI Max+ 395 PC with 94 GB RAM; UD-Q2_K_XL with the Q5_K
attention pack, through the server):

| | Without the drafter | DSpark |
| --- | :---: | :---: |
| 6 swe-bench coding tasks (sampled, end to end), no iGPU lane | 48.4 tok/s (v1.3.42) | 61.9 tok/s |
| The same with the iGPU lane | 51.7 tok/s (v1.3.41) | 67.3 tok/s |
| 1,024-token answers, code / prose (greedy, no iGPU lane) | 46.6-48.6 / 47.2-47.5 tok/s | 63.4-66.4 / 50.5-51.4 tok/s |
| Prompt 8K / 16K | 850 / 978-1,028 tok/s | 799 / 977-1,000 tok/s |

Per swe-bench task the drafter gives +16-32%. On these tasks 68-82% of the drafts are accepted and a step moves 2.7-3.7
tokens (4.7 on a task that copies code); prose gains least (~2 tokens a step). A window's column costs ~9 ms here,
about what the CPU lane needs for one token's RAM-tier experts (8.4 ms: ~93 experts at the RAM's ~93 GB/s), so the
window, not the drafter, sets the speed - a faster CPU / RAM or an iGPU lane helps more than a faster GPU. NVIDIA: on a
rented DGX Spark (v1.3.114) decode went 21.0 -> 22.8 tok/s with 78% of the drafts accepted.

**How a step is chosen.** When the last 3 tokens occurred earlier in the prompt or the answer, their continuation
(up to 5 tokens) is the window's drafts and the drafter does not run - while those repeats keep being right: every
step checks the repeat's proposal against the tokens that came, a share right per position. Below 0.75 at the first
position the drafter takes over; otherwise the repeat's drafts are used while the product of the shares stays at 0.5
or above. With no confident draft the token runs alone. The engine log's `ds4 dspark:` line after each answer counts
the windows, the drafts and the acceptance, and the tokens a step.

**Settings** (the config's `env`):

| Setting | Default | |
| --- | --- | --- |
| `STRATA_DS4_DSPARK` | off | `<the drafter GGUF>`: load DeepSeek's DSpark drafter and decode with it |
| `STRATA_DS4_DSPARK_THR` | 0.6 | the confidence a draft needs; the window stops at the first one below (swe-bench: 0.7 61.3, 0.8 59.8 tok/s against 0.6's 61.9; ds4 uses 0.7) |
| `STRATA_DS4_DSPARK_K` | 5 | drafts a window at most (1-5) |
| `STRATA_DS4_DSPARK_CHAIN` | off | `<p>`: keep drafts while the product of their confidences stays >= p, instead of each one >= the threshold |
| `STRATA_DS4_DSPARK_NGRAM` | on | repeats from the text first (above); `0`: the drafter at every step |
| `STRATA_DS4_DSPARK_NGRAM_P` / `_NGRAM_CHAIN` | 0.75 / 0.5 | a repeat is used while its first position is right this often; its length while the product of the positions' shares stays at this |
| `STRATA_DS4_DSPARK_EXPERTS` | 256 (all) | the drafter's expert slots in VRAM a stage (8-256; 2.1 GB a stage at 256): an expert without a slot is left out of its row and the most used are copied in after each draft. 128 / 64 give 3-5 GB back for 2 more points of VRAM hits but make the drafts 30-50% slower (an option for a 24 GB card, not measured there) |
| `STRATA_DS4_DSPARK_UPLOADS` | 4 | experts a stage copied into the cache after a draft (0-8) |
| `STRATA_DS4_DSPARK_PROF` | off | `1`: the drafter's GPU time by phase in the report at exit |
| `STRATA_DS4_DSPARK_LOG` | off | `<n>`: every n windows, their mean time and width (stderr) |

## Uncensoring at run time (`--uncensor`)

From v1.3.71 the engine can take a model's refusal direction out while it runs, switchable per request, with a
16-82 KB file made from the model itself (`--abl-build`). How to turn it on, make and check the file, what it measured
and the older ways are all in [UNCENSOR.md](UNCENSOR.md).

## Settings

(The config file itself - where it is and every key - is described in [SETTINGS.md](SETTINGS.md).)

Environment variables for the engine, in the config's `"env"` block (all of Maya's are listed in
[docs/maya/README.md](maya/README.md#tuning); these matter most here):

| Setting | Default | |
| --- | --- | --- |
| `STRATA_GLM_WARM` | on | `0`: no warm-up at start (the tiers fill as experts are used; the first answers are slower) |
| `STRATA_GLM_RAM_GB` | from free RAM | a fixed size of the pinned RAM tier in GB. A larger tier cuts SSD waits but gives the CPU lane those experts: on a DDR4 desktop with V4.1, 80 -> 88 GB cut SSD waits 11.1 -> 8.7 ms a token, the lane rose 9.4 -> 10.6 ms, decode +3% (docs/BENCH) |
| `STRATA_GLM_EXPERT_HELPER` / `_GIB` | off / 13 GiB | DeepSeek V4.1, CUDA: a separate visible GPU index (e.g. `1`) and its resident Decode-expert pool budget. `0` disables it. Prefill stays on the primary; absent hardware, a pool that cannot fit, or a helper error retains CPU execution. Requires an unsplit native pack and CPU lane, with layer graphs off; DSpark verify windows use it too (v1.3.94). [Configuration and limits](EXPERT_HELPER.md) |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 (16 with unified memory) | RAM left free for the system when the RAM tier is sized; the AMD setup writes 16 into its configs |
| `STRATA_GLM_RAM_FRAC` | 0.73 | Linux, a discrete GPU, no `STRATA_GLM_RAM_GB`: the RAM tier pins at most the larger of this share of the RAM and the RAM less `STRATA_GLM_RAM_KEEP_GB` (on machines of 100 GiB of RAM or more; smaller ones keep the share alone, as before). More than ~73% pinned thrashed a 94 GB machine. Set on its own, the share alone caps the tier (the behaviour before the keep). Held to 0.05..0.95 (a value that would lift the cap is clamped, with a line in the log) |
| `STRATA_GLM_RAM_KEEP_GB` | 24 | Linux, a discrete GPU, no `STRATA_GLM_RAM_GB`: RAM kept out of the RAM tier, counted from the whole RAM, for the page cache (the shards, V4.1's Engram tables), the server and the desktop - a need that does not grow with the RAM, so machines with 100 GiB of RAM or more pin more than 73%: 123.5 GiB of RAM caps the tier at 99.5 GB instead of 90.2 (DeepSeek V4.1 UD-IQ2_XXS on an RTX 5090: 30.2 -> 34.5 tok/s), 251 GiB at 227. 64 and 96 GB machines (62.5 / 94 GiB) keep the 73% cap (45.6 / 68.6 GB); set the variable to use the keep there too. If other programs grow by more than ~8 GB after the engine starts (a heavy desktop, a browser with many tabs), set `32`. The tier also stays below the free RAM less `STRATA_GLM_RAM_HEADROOM_GB`. The `RAM tier ... pinned` line in the log says which bound set the size |
| `STRATA_GLM_STREAM_FETCH` | on | an expert read from the SSD is copied to VRAM chunk by chunk as the reads land, the PCIe copy running behind the SSD (v1.3.31; DeepSeek V4.1 -9% a token); `0`: the whole expert after the read |
| `STRATA_GLM_RAM_THP` | on (Linux) | the RAM tier in transparent huge pages, as far as the free 2 MB blocks go (an APU lane re-validates 512x fewer pages: no multi-second lane stalls); `0`: the driver's pinned pages |
| `STRATA_GLM_RESERVE_MB` | 1536; 700 on a CUDA card of 12 GB or less and for Qwen3.8 on any CUDA card | VRAM left free beside the expert pool. Qwen3.8 on CUDA keeps 700 on every card from v1.3.132 (no DSpark, and its prompts borrow from the pool): RTX 5090, 432 more expert slots, 4K + 512 decode 113.9 -> 121.1 tok/s without a drafter, 167.7 -> 175.7 with MTP. CUDA keeps 1536 too from v1.3.82 on cards over 12 GB: an RTX 5090 running DeepSeek V4.1 at `--max-context 65536` ran out of VRAM in the attention's and the DSpark drafter's buffers with 700 kept. HIP keeps more since v1.3.68: the first prompt's hipBLASLt GEMMs load their kernels then (~0.7 GB), and with 0.7 GB kept an R9700 at `--max-context 131072` could not load them - every prompt GEMM failed and an 8K prompt took 1,300 s. Decode ~3% slower for the smaller pool. The engine flag `--vram-reserve-mib N` (in a config's `args`) sets the same reserve on every model - the GLM-5.3 / Qwen3.8 / DeepSeek V4.1 pool and the older qwen4exp path's `--expert-cache auto` alike (up to v1.3.129 that pool ignored the flag, #39). With both set, the flag wins (setup writes this variable as a machine default, the flag is the config's own choice; the log then says `NOTE: --vram-reserve-mib N overrides STRATA_GLM_RESERVE_MB=M`) |
| `--kv q4_0` / `k8v4` / `int8` (an engine flag, in a config's `args`) | `fp16` | Ornith-1.5 122B / 397B (`qwen35moe`, from v1.3.134): the attention layers' K/V cache in fewer bytes, and the VRAM it frees goes to the expert pool. `q4_0`: 4-bit codes after a Hadamard rotation (28% of fp16), `k8v4`: 8-bit K + 4-bit V (41%), `int8`: 8-bit (53%). At `--max-context 131072` Ornith-1.5-397B's cache is 4.00 GB in fp16, 1.12 / 1.62 / 2.12 GB in q4_0 / k8v4 / int8 - on a 16 GB card fp16 stops with `not enough VRAM for the expert pool` while all three start (#44); Ornith-1.5-122B with its NextN drafter 3.25 -> 0.91 GB (q4_0). KL against fp16 (122B 24G64G, 511 scored tokens after an 8K / 32K context): q4_0 0.019 / 0.007 (same top token 93.9% / 98.8%), k8v4 0.013 / 0.005, int8 0.008 / 0.003; after an 8K Japanese context q4_0 0.031 (same top token 91.6%), k8v4 0.021, int8 0.014, nll within 0.005 of fp16. Speed (RTX 5090, 122B 24G64G, NextN drafter): at 32 GB decode the same as fp16 at 4K-64K, prompts 5-9% slower at 32K-64K; capped at 16 GB with a 128K context, q4_0 leaves 7.05 GB for experts instead of 4.71 - a 32K prompt 746 -> 1,654 tok/s, decode 26.3 -> 34.9 tok/s. Pick the lightest that fits: `int8` when it does, `q4_0` when VRAM is tightest. Qwen3.8's older path has had these since PR #21; GLM-5.3 takes `int8` (its latent cache); qwen4exp's fast path keeps fp16 |
| `STRATA_GLM_CPU_LANE` | one thread per physical core (an integrated NVIDIA GPU, DGX Spark: off) | CPU threads computing RAM-tier experts; `0` = off |
| `STRATA_GLM_CPU_EXACT` | `1` (Qwen3.8 on x86 CPUs), `0` (other models: their rows were slower - GLM-5.3 decode -14%, V4.1 -5% on an RTX 5090) | the CPU lane computes its experts with the GPU kernels' own arithmetic (the GPU's 8-bit activations, its integer sums), so an expert gives the same numbers on either side; `0` = the older ggml-cpu dots. ARM CPUs keep the older dots |
| `STRATA_GLM_LANE_XC` | `1` (CUDA on x86, Qwen3.8), `0` (other models, HIP, ARM) | with `STRATA_GLM_CPU_EXACT`, the CPU lane's experts give the GPU's bits exactly (the GPU's float order and its exp in the SiLU), and the layer's expert sum is taken in fixed point (each weighted expert row to a 64-bit integer of 2^-34, the sum rounded to a float once: the same bits in any order, at least as accurate as the float sum), so it does not depend on which experts the CPU computed. The prompt's CPU share follows too (`STRATA_GLM_PREFILL_XROWS`). Greedy answers are then the same run to run without `STRATA_DETERMINISTIC` (prompts up to about 2,000 tokens a chunk on Qwen3.8; longer chunks are added into the output window as it fills, in the order the experts ran). `0` = the CPU's experts summed apart and added last, the prompt's share by timing (the older behaviour). Off with expert deferral (`STRATA_GLM_DEFER`) and EXL3 experts. Other models (GLM, Ornith, DeepSeek V4): `1` turns it on (not measured yet) |
| `STRATA_Q4X_VERIFY_EXACT` | `1` (CUDA), `0` (HIP) | Qwen3.8 with drafts (MTP, DFlash 2): the verify window's dense GEMVs give each column the token path's sums, so greedy answers with drafts equal the ones without (v1.3.125). On HIP that runs as one GEMV a column - the dense weights and the output head read once per column - and cost an AI PRO R9700 with MTP 87 -> 67 tok/s (#42), so HIP keeps the shared-read GEMVs (as before v1.3.125; answers with drafts can part from the ones without late in an answer). `1` / `0` force either; `STRATA_DETERMINISTIC` sets `1` |
| `STRATA_GLM_CPU_CAL_RETRY` | 3 | when the start-up timing of the CPU lane looks disturbed (other work running on a quarter of the CPUs, or its slowest run over 3x the median), it is taken again up to this many times, 1 s apart, and the lowest kept; a CPU still busy after that is reported in the log. `0`: one timing |
| `STRATA_GLM_TAIL_ON_TOP` | 1 | when VRAM holds every expert, the spares and the prompt path's borrowed tail, the tail is allocated on top of the experts' slots, so a prompt evicts none. With unified memory (DGX Spark) too from v1.3.119, where the shared memory has the room (Qwen3.8 ISTA IQ3_XXS: the decode after a 32K prompt no longer re-reads experts from the SSD). `0`: the tail is cut from the experts' slots (the layout before v1.3.104; on unified memory, before v1.3.119) |
| `STRATA_GLM_X3_SPLIT` | off | an EXL3 pack: the CPU threads compute this fraction of each route's RAM-tier experts while the iGPU computes the rest (`auto`: by their timings); slower on a Ryzen AI Max+ 395 (shared power and heat), see [EXL3](exl3/README.md#settings) |
| `STRATA_GLM_EXL3_DECODE` | by GPU | an EXL3 pack: `new` (tensor-core tiles, RTX 30+) or `legacy` (CUDA cores) on any GPU |
| `STRATA_GLM_CPU_PIN` | off | `<CPU list>` (e.g. `0-15`, `0,2,4-7`): pin the CPU lane's workers to those CPUs and cap their count at the list's size, one GPU or a split, Linux and Windows (v1.3.29); `1` / `0`: a split's per-part pools pinned or not, as before |
| `STRATA_GLM_PCIE_SHARE` | measured at start | the share of RAM-tier experts copied over PCIe instead of computed on the CPU (`0` = all on the CPU, `1` = none) |
| `STRATA_GLM_PCIE_SHARE_LATE` | off | `<share>[:<steps>]`: each request starts on the share above and moves to this one after `<steps>` token steps (default 256). A lower share copies fewer experts over PCIe and so refills the GPU's pool more slowly; this lets the start of each answer refill it first (issue #14). `<share>:settle[:<window>[:<gain>[:<drop>]]]` switches once per session instead: every `<window>` token steps (default 128) it measures the VRAM hit rate, moves to this share when a window gains less than `<gain>` points (default 1) on the one before, and goes back when it falls `<drop>` points (default 5) below that level (a new topic) - the warm pool builds over several answers, which a per-request switch never reached. Keep it only if a plain `--bench` gets faster |
| `STRATA_GLM_LANE_LEAN` | 4 | when the CPU lane does an expert this many times faster than the PCIe link, every RAM-tier expert goes to the CPU once the pool has settled (the `:settle` switch above, automatically), the measured split until then and when the VRAM hits fall. `0`: the measured split only. A share or plan set by hand wins ([Small VRAM](SMALL-VRAM.md)) |
| `STRATA_GLM_POOL_FLOOR` | 4 | the expert pool's smallest size in slots a layer (the spares and one resident); below it the engine refuses to start and says how much VRAM is short. Before v1.3.82 a route's experts (8) were the floor; a pool between the two runs the rest of every route from the scratch slots and the CPU lane ([Small VRAM](SMALL-VRAM.md)) |
| `STRATA_GLM_PREFILL_KEEP` | 4 | slots a layer the pool keeps while a prompt runs, on a pool that could lend no prompt chunk while keeping the full amount (a route, the spares and 2); the rest is lent to the prompt path and refilled after ([Small VRAM](SMALL-VRAM.md)) |
| `STRATA_GLM_PREFILL` / `--prefill` | `auto` | the prompt chunk: `auto`, a number of tokens, or `0` (token by token) |
| `STRATA_WATCHDOG_S` | 180 | the serve watchdog: a request with no progress for this many seconds ends the engine, and the server starts it again (`0` = off). A big file copy while a long prompt runs (memory pressure: 200 GB to a network share here) can hold the engine up for minutes at one place, which the watchdog reads as stuck; raise it (`600`) or copy at another time |
| `STRATA_GLM_PREFILL_SUB` | auto (NVIDIA), 1024 (HIP) | the prompt's mixer and dense-FFN sub-batch (16-8192). HIP's default 1024 since v1.3.40 (it was 256: AI PRO R9700 DeepSeek V4 8K / 16K prompts +8% / +10%, GLM-5.3-Flash +13% / +15%) |
| `STRATA_GLM_PROMPT_SKIP` | on | DeepSeek V4.1, one device: a long prompt's rows before its last 2,668 stop at layer 20's attention (layers 20-39 read them only through layer 20's compressed KV; exact up to the experts' summing order; v1.3.58, 32K prompts 64 -> 101 tok/s on an AI PRO R9700). `0`: off |
| `STRATA_GLM_V41_BATCH` | on | DeepSeek V4.1's prompt attention: its projections (q_a, kv, the compressor, q_b, the output LoRA) as GEMMs over a sub-batch, the ring / compressed rows / indexer / attention still a token at a time (v1.3.59; with the skip, AI PRO R9700 8K / 16K / 32K / 60K prompts 86 / 101 / 101 / 101 -> 185 / 242 / 273 / 262 tok/s, KL at the run-to-run floor). `0`: the token path's attention a token at a time |
| `STRATA_GLM_V41_ATTN_TOK` | off | `1`: the V4.1 prompt attention's window ring, compressed rows, indexer and attention a token at a time again (v1.3.59's path, for A/B). Off: batched over each sub-batch (v1.3.60; with the Engram GEMM below, AI PRO R9700 + iGPU, UD-IQ2_XXS 8K / 16K / 32K / 60K prompts 237 / 292 / 314 / 277 -> 363 / 596 / 781 / 896 tok/s, Maya-XS 192 / 334 / 327 / 288 -> 964 / 1055 / 1093 / 1028; KL 0.0038-0.0040 at 8K-40K against the token loop) |
| `STRATA_GLM_V41_ENG_GEMM` | on | V4.1 prompts: Engram's engram_kv as one FP16 GEMM over each 256-token block (v1.3.60; the token path's GEMV read its ~160 MB again for every token, 32K prompt 20 -> 3.9 s). `0`: a token at a time |
| `STRATA_GLM_V41_CAND_ROW` | off | `1`: V4.1 prompts' candidate blocks (the indexer's rows past 16K visible) a row at a time again. Off: a launch a group of rows (v1.3.63). On HIP gfx11 / gfx12 V4.1's 32-head indexer scores on WMMA too (`STRATA_DS4_IDX=1` goes back; 40K prompt 3.4 -> 1.9 s of GPU time); with both, AI PRO R9700 Maya-XS 32K / 60K prompts 1,093 / 1,028 -> 1,119 / 1,214 tok/s (UD-IQ2_XXS 60K: the SSD is the limit, unchanged) |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | your expert usage between starts; `0` = off |
| `STRATA_GLM_PRIOR` | set by the server | a routing profile (`expert_counts.txt` lines, `tools/glm_expert_prior.py`) for a pack without its own: the tiers warm in its order and each layer's VRAM share follows it. The server points it at `data/expert_counts/` for the models Abyss ships one for (Qwen3.8-Flash-Next, any quant; v1.3.86: RTX 5090, ISTA IQ3_S, `--bench` 87.6 -> 116 tok/s with the swap rule above) |
| `STRATA_GLM_NO_REUSE` | off | `1`: no conversation reuse |
| `STRATA_GLM_THINK_RESERVE` | 1024 | (server) tokens a thinking block always leaves for the answer (at most half of the request's `max_tokens`) |
| `STRATA_GLM_END_GUARD` | 1 | (server) the end-of-turn guard: a turn that ends inside the reasoning closes the reasoning instead, and a turn that ends right after it with no answer takes the next most likely token, once each (a heavily quantized model's empty answers); `0` = off |
| `STRATA_GLM_THINK_BRIDGE` | (a sentence) | (server) what a reasoning block closed by the thinking budget or the loop guard says just before its `</think>`: that the thinking stops here and the answer follows in the question's language (a bare `</think>` mid-thought let Qwen3.8 go on thinking in English in the answer, issue #36); another text replaces it, `0` = a bare `</think>` as before |
| `STRATA_ENGINE_STALL_S` | 90 | (server) an engine silent that long with no CPU, disk or GPU work is ended and started again by the next request; `0` off |
| `STRATA_ENGINE_START_S` | 1800 | (server) the most a start may take before READY (a model loading from the SSD takes minutes); past it the engine is ended and the start fails with a line; `0` no limit (v1.3.78) |
| `STRATA_ENGINE_DRAIN_S` | 120 | (server) a request stopped early (the client left) is drained to its DONE; an engine silent that long after the STOP is ended; `0` no limit (v1.3.78) |
| `STRATA_ENGINE_WRAP_S` | 40 | (server) at Ctrl+C / SIGTERM a request in progress closes its thinking and answers for up to this long, then is cut |
| `STRATA_ENGINE_QUIT_S` | 50 | (server) how long the engine gets to end after QUIT (with `STRATA_GLM_SLOT_KEEP` it writes its conversation first); then it is killed. llama-swap's unloadTimeout should exceed it |
| `STRATA_VISION_START_S` / `STRATA_VISION_ENCODE_S` | 600 / 120 | (server) the image encoder's start and each picture's answer; past them it is ended, the request gets an error and the next picture starts it again (v1.3.78) |
| `STRATA_HTTP_BACKLOG` | 256 | (server) the listen backlog: requests waiting to be accepted |
| `STRATA_MAX_BODY_MIB` | 256 | (server) the largest request body; a larger one is answered 413 before it is read |
| `GPU_MAX_HW_QUEUES` | 2 (HIP) | ROCm's hardware queues a device: the server passes 2 when neither the shell nor the config sets it (v1.3.40). AI PRO R9700 DeepSeek V4 prompts 8K / 16K: 586-650 / 676-684 tok/s on 1 (a value R9700 guides spread against a MES firmware's idle power; 12 W idle here either way), 653-737 / 748-756 on ROCm's 4, 866-995 / 999-1,011 on 2 |
| `STRATA_GLM_IGPU` | auto | the iGPU lane: on when `libabyss_igpu.so` and a HIP device are there and it is clearly faster than the CPU (`STRATA_GLM_IGPU_MARGIN`); `0` off, `1` on even when slower. `STRATA_GLM_IGPU_LIB=<path>`: another library |
| `STRATA_GLM_IGPU_MARGIN` | 1.5 | auto: the iGPU takes the lane only when its start-up timing of an expert is this many times faster than the CPU's (the timing leaves out what an iGPU call costs inside a real token: on a Ryzen AI Max+ 395 + RTX 5090, Qwen3.8 16G64G timed a near tie - 0.022 ms vs 0.019-0.025 - and every run that picked the iGPU decoded slower, MTP 141 -> 132 tok/s, default 94 -> 89; GLM-5.3 / V4 / V4.1 time the iGPU 1.9-2.7x faster and keep it). With the exact combine (`STRATA_GLM_LANE_XC`, Qwen3.8 by default) the CPU keeps the lane: the iGPU's sums are not the GPU's bits and the answers would depend on the lane again. `1` = the faster of the two (the older pick) |
| `STRATA_GLM_IGPU_TIMEOUT_MS` | 20 | a lane call later than this goes to the CPU threads (0: wait) |
| `STRATA_GLM_IGPU_POKE_MS` | 500 | how long after a route the service thread keeps the iGPU busy with a no-op kernel every 50 us (v1.3.40; was 20). DeepSeek V4.1 waits on the SSD between routes for longer than 20 ms, and an idle iGPU then took 21-27 ms to answer (AI PRO R9700 + 8060S, the slowest held-out sequence: 123 late calls and 80.4 ms a token; 500: 3 calls, 65.6 ms) |
| `STRATA_GLM_IGPU_COLS` | on | a verify window's (DSpark / repeat drafts) iGPU-lane experts in one lane call for all its columns, each expert read once for every column that routes to it (v1.3.89; the same bits; RTX 5090 + Radeon 8060S, DeepSeek V4.1 UD-IQ2_XXS + DSpark: lane 60-62 -> 54-57 us an expert, decode -3-5% a token, swe-bench 48.6 -> 51.0 tok/s). `ABYSS_IGPU_COLS_UNION=0`: the one call keeps per-column grids (no shared reads); `0`: a lane call a column |
| `STRATA_GLM_IGPU_EARLY` | on | a route that reads experts from the SSD starts its iGPU lane before the reads instead of after them (v1.3.36; DeepSeek V4.1 -2%, same logits); `0`: after |
| `STRATA_GLM_IGPU_XQ` | on | the lane takes the token's input already quantized (q8_1) by the GPU from the request instead of quantizing it again (v1.3.36, bit-identical); `0`: quantized on the iGPU |
| `STRATA_GLM_PREFILL_IGPU` | on | the prompt's RAM-tier experts on the iGPU; `0`: the CPU threads |
| `STRATA_GLM_PREFILL_CPU` | on | the prompt's CPU share (RAM-tier experts computed on the host while PCIe stages the rest); `0`: every expert over PCIe |
| `STRATA_GLM_PREFILL_XROWS` | 32 | with `STRATA_GLM_LANE_XC`: the prompt's experts of at most this many rows of a chunk are computed in the decode kernels' arithmetic - on the GPU when resident, on the CPU from the RAM tier (all of those, whatever the timing), the same bits either way - and the CPU takes no larger expert; larger ones go through MMQ. So the prompt does not depend on what was resident or on the CPU's learned share (an expert read from disk still goes through MMQ). `0`: the timing-learned share (1-32) |
| `STRATA_GLM_PREFILL_PIN_MB` | none | the most MB one of the CPU share's pinned host buffers may take; when they do not fit (or do not allocate, or would leave under 2 GB of RAM free) the prompts stage every expert over PCIe from then on (#31). `0` = as `STRATA_GLM_PREFILL_CPU=0` |
| `STRATA_GLM_IGPU_FUSED` | on | the prompt's iGPU experts through the fused int8 tensor-core pass (gfx1151, IQ2_XS / IQ3_XXS experts at 4096 / 2048; v1.3.18); `0`: the one-row kernels. `ABYSS_IGPU_FUSED_CHECK=<n>` compares the first n calls against them (stderr) |
| `STRATA_GLM_MV_Q6K_SPLIT` | per GPU | `1`: Q6_K GEMVs of up to 4,096 blocks 4 warps a row, `0`: one warp a row. Unset: the engine's per-GPU choice (gfx12 one warp - an R9700 7.5 instead of 8.8 ms a DeepSeek V4.1 token -, others 4 warps); `./abyss.sh --gpu-tune` measures it (v1.3.51) |
| `STRATA_GLM_MV_SPLIT` | auto | dense GEMVs with a row over 4 warps: unset = under 256 blocks, and Q6_K up to 4,096 blocks (v1.3.18); `0` one warp a row always, `2` / `4` that split always. Not bit-identical to one warp a row |
| `STRATA_GLM_MV_FUSE` | on (HIP) | a small GEMV call of several types (DeepSeek V4's projections of x) in one launch, 4 waves a row (v1.3.40; a small GEMV costs 15-29 us on an AI PRO R9700 whatever its size); `0`: a launch per type. Not bit-identical to the typed kernels |
| `STRATA_GLM_GRID_PAD` | on (HIP, except gfx11) | launches whose waves land at or just below a multiple of the GPU's wave slots get a few empty blocks (v1.3.40): on gfx1201 such a dispatch delays the next one by 8-14 us (ROCm/TheRock#9300), and DeepSeek V4's 4096-row GEMVs and expert down hit it ~275 times a token (AI PRO R9700: 22.2 -> 20.2 ms a token). Bit-identical. Off by default on gfx11 (issue #9, RX 7900 XT: no stall, padding within noise; v1.3.54); `0` off everywhere, `1` on everywhere |
| `STRATA_GLM_HC_CPU` | on (HIP), off (NVIDIA) | deepseek4: the CPU / iGPU lane's part of a MoE layer is added by the next layer's attention-side mHC read (hc3) instead of a wait kernel of its own - a launch less a layer, the answer read by hc3's 32 blocks (v1.3.42). Bit-identical; `0`: the wait kernel; `1` on NVIDIA too (not measured there) |
| `STRATA_Q4X_PLE_GGUF` | the pack's | Qwen3.8-Flash-Next: read the PLE table (`per_layer_token_embd`) from this GGUF instead. IQ4_NL, Q5_1, Q8_0 or BF16 (Q5_1 / BF16 from v1.3.117). RTX 5090, 16G64G (M16b) KL against NVFP4: IQ4_NL 0.0826, Q5_1 0.0732, Q8_0 0.0714 at the same speed; the table is read row by row, so a bigger type costs disk, not VRAM |
| `STRATA_Q4X_PLE_PREFETCH` | on | Qwen3.8-Flash-Next with the MTP drafter: each verify window's PLE table rows (16 SSD reads a token) are read into the row cache as soon as the drafter knows the token, a draft step before the window needs them (v1.3.87). RTX 5090, ISTA IQ3_S: the window's wait for its rows 0.8-1.1 ms -> 0.1-0.25 ms, drafter on +3%. `0`: read when the window starts |
| `STRATA_Q4X_HC_FUSE` | on (NVIDIA), off (HIP) | Qwen3.8-Flash-Next: a token's hyper-connection mixes (97 a token) in two launches instead of five, the same bits (v1.3.86; `hc4_fused_parity` checks it). RTX 5090, ISTA IQ3_S: the mixes 2.98 -> 2.21 ms a token, token by token +8%. The MTP verify windows too from v1.3.87 (a window's 97 mixes 2.6 -> 1.8 ms at 2 columns, 5.5 -> 2.6 ms at 5; drafter on +8%). `0`: five launches; `2`: tokens fused, windows five (for A/B) |
| `STRATA_Q4X_GDN_FUSE` | on (NVIDIA), off (HIP) | Qwen3.8-Flash-Next: a delta-net layer's middle (the q / k L2 norm and the gates, the recurrence, the gated output norm and its q8_1) in one launch instead of four, token by token and in the MTP verify windows (there after the gates: three launches into one); the same bits (`gdn_core_parity`). RTX 5090: 36 layers 0.44 -> 0.33 ms a token alone, about +1% decode (within the bench's drift). `0`: the separate launches |
| `STRATA_Q4X_QSA_BATCH` | on (NVIDIA), off (HIP) | Qwen3.8-Flash-Next prompts: the QSA layers' attention for the rows past the first 2,051 cells runs 64 rows a launch, each over its own selected cells read in place, instead of a cell copy and an attention launch a row; the indexer keys of a sub-batch go in two launches instead of one a row. The same bits (`qsa_rows_parity` checks both). RTX 5090, ISTA IQ3_S: the QSA stage 2,000 -> 346 ms over the bench's 12,353 prompt tokens; 8K prompt 1,555 -> 2,360 tok/s (+52%), 32K 1,710 -> 3,015 (+76%), 2K +5%. `0`: a row at a time |
| `STRATA_Q4X_QSA_DECODE` | on (NVIDIA), off (HIP) | Qwen3.8-Flash-Next decode: the QSA layers' attention token by token and in the MTP verify windows reads its cells in place, a few query heads a block, instead of a cell copy and attn_decode (a window's columns in one launch, one selection call for them, its indexer keys in two launches). The same bits (`qsa_rows_parity`). RTX 5090, 2,051 cells a layer: a token 22 -> 18 us, a 4-column window 88 -> 24 us, 8 columns 177 -> 34 us; --bench decode (drafter on) 126.9 / 128.9 -> 131.0 / 131.3 tok/s (about +2.6%, inside the bench's drift). `STRATA_Q35_ROWS_HPW` (query heads a block: 1, 2, 3, 4 or 6; default 2 for one row, 4 up to 5, else 6) is for tuning. `0`: the old launches |
| `STRATA_GLM_WAIT_FETCH` | on (HIP), off (NVIDIA) | the MoE layer's wait for disk reads and its copies of RAM-tier experts in one launch instead of two (v1.3.42; the same work); `0`: two launches |
| `STRATA_HIP_EXPERTS_LEGACY` | off | `1`: the original decode expert kernels instead of the RDNA ones (gfx1100, gfx1201 since v1.3.40: gate/up and down ~20% faster on an AI PRO R9700; gfx1151 all but the IQ3_XXS down). Bit-identical either way |
| `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN` | 4 / 16 (Qwen3.8: 2 / 8) | a RAM-tier expert replaces a resident one when its use count passes ratio x the resident's + margin (2 / 4: the old, churnier rule). Qwen3.8-Flash-Next swaps sooner and moves up to 16 experts a token boundary (`STRATA_GLM_PROMOTE`, else 8): its ~2 MB experts copy in ~0.3 ms (v1.3.86; RTX 5090, ISTA IQ3_S: VRAM hits 60-77% -> 68-89%, +12-15%) |
| `STRATA_GLM_BG_SPLIT` | 1 for Qwen3.8, else 0 | background promotions also while the CPU lane takes only part of the RAM-tier experts (0: only when it takes all of them, as before). Qwen3.8 whose VRAM pool holds half of the experts or more swaps at ratio 1 / margin 2 (RTX 5090, Swift 1.5 IQ2_XS 32 GB: VRAM hits 95.0 -> 96.2%, 176 -> 182 tok/s); a tighter pool keeps 2 / 8 |
| `STRATA_GLM_FETCH_COLS` / `STRATA_GLM_CPU_WAIT_COLS` | 1 for Qwen3.8, else 0 | the speculative verify window waits for the CPU lane and fetches its experts with one kernel for all columns instead of one per column (the same arithmetic) |
| `STRATA_GLM_GRAPH` / `STRATA_GLM_GRAPH_MOE` | 1 / 1 on CUDA, 0 on HIP | Qwen3.8's verify window replays each delta-net layer from a CUDA graph (0: launched kernel by kernel; `_MOE=0`: the MoE dispatch outside the graph). The same tokens; launches a token 453 -> 254 (RTX 5090: no change in speed; DGX Spark, with PDL: 43.4 -> 47.1 tok/s) |
| `STRATA_GLM_PDL` / `STRATA_GLM_PDL_PREFETCH` | on for an integrated GPU (sm_90+) and for Qwen3.8 on a discrete sm_120+ card (RTX 50, v1.3.132); off for other discrete cards and models / 1 | programmatic dependent launch: a kernel starts while the one before finishes (DGX Spark, Qwen3.8 16G64G: 25.0 -> 30.7 tok/s without MTP, 43.4 -> 46.5 with it; RTX 5090, Qwen3.8 UD 3.1 4K + 512: 128.0 -> 137.8 without MTP, 178.6 -> 184.7 with it; GLM-5.3 / V4 / V4.1 on an RTX 5090 measured no gain and keep it off). `_PREFETCH`: the GEMVs pull their weight rows into L2 early (0: not; 2: also without PDL) |
| `STRATA_GLM_UMA` / `STRATA_GLM_UMA_HEADROOM_GB` | 1 / 8 (CUDA) | an integrated GPU (DGX Spark) sizes its tiers from the unified memory, leaving `_HEADROOM_GB` free (0: the discrete-card policy, A/B) |
| `STRATA_GLM_UMA_SHARE` / `STRATA_GLM_UMA_OTHERS_GB` | 0 / 16 | an integrated GPU: when other processes hold more than `_OTHERS_GB` of the shared memory and the experts no longer fit because of it, the engine stops at load and names the biggest ones (a second engine left running ran a DGX Spark out of memory); `STRATA_GLM_UMA_SHARE=1` starts anyway. A model bigger than the machine with nothing else running is not stopped |
| `STRATA_GLM_UMA_DIRECT` | 0 | an integrated GPU reads RAM-tier experts in place instead of copying them into VRAM slots. Off by default: it stops promotions, so the hit rate stays low (DGX Spark, Qwen3.8 16G64G at 32 GB: copy 45.8, direct 34.9 tok/s) |
| `STRATA_GLM_DISK_BUFFERED` | 0 | (test) expert reads from the SSD through the page cache instead of O_DIRECT |
| `STRATA_GLM_BG_PACE_KB` | 1024; 512 for Qwen3.8 on CUDA | the tier moves' copies (the refill after a prompt, the swaps) go in pieces of up to this many KB a MoE layer, while the next layer's mixer runs, instead of all at the token boundary - there they held up the next token's host-mapped reads until they had landed (after an 8K prompt ~10 tokens at 35-39 ms). Qwen3.8 (~2 MB experts) on CUDA uses 512 from v1.3.132: a 1 MiB piece still held up the next layer's route writes (RTX 5090, UD-IQ3_XXS: 110 -> 118 tok/s); GLM-5.3 / V4 / V4.1 (7-8 MB experts) keep 1024 (512 was no faster there and undid the decayed counts' gain). `0`: at the boundary, as before |
| `STRATA_GLM_BG_DECAY` / `STRATA_GLM_BG_DECAY_N` / `STRATA_GLM_BG_DECAY_MARGIN` | 0.995 / 8 / 2 for Qwen3.8, GLM-5.3 and DeepSeek V4 / V4.1 on CUDA (Qwen3.8 only on HIP); else 0 | the background tier moves (v1.3.132) rank experts by route counts that decay by this factor every token boundary (about 200 tokens of memory) instead of the halved integer counts: a RAM-tier expert replaces its layer's coldest resident when its count passes that one's by `_MARGIN`, at most `_N` swaps a boundary. It moves weights only - the answers are the same (deterministic md5 unchanged). RTX 5090, 4K + 512, iGPU off: Qwen3.8 UD 3.1 UD-IQ3_XXS 128.5 -> 136.1 tok/s (MTP 185.5 -> 196.6), DeepSeek V4 0731 UD-Q2_K_XL 53.7 -> 58.2, GLM-5.3 Maya-S-v2 IQ2_XXS 43.8 -> 46.0, V4.1 Compact JA256 47.3 -> 49.0. `0`: the old counts with `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN` |
| `STRATA_GLM_BG_LANE` | `512:128:1` for GLM-5.3 and DeepSeek V4 / V4.1 on CUDA; else off | `<KiB a lane expert>:<KiB off>[:<kinds>[:<max KiB>]]`: when the CPU lane takes a route's ne RAM-tier experts, its service thread lets ne x KiB - off of the queued tier copies (kinds 1 promotions, 2 demotions, 3 both) go at once on a second copy stream - they cross the link while the CPU computes, where it was idle (an nsys trace of V4 0731: ~215 us a layer, 96% idle), and end before the device reads the answer. It moves weights only (deterministic md5 and the exact-lane md5 with moves on unchanged). RTX 5090 + Gen4 x4, 4K + 512, iGPU off: V4 0731 UD-Q2_K_XL 58.3 -> 60.7 tok/s, GLM-5.3 Maya-S-v2 IQ2_XXS 46.8 -> 49.1, V4.1 Compact JA256 50.0 -> 52.2 (+4-8% over five sets of runs). Not for Qwen3.8 (its ~30 us lane experts leave no room: -7.5%). `0`: off |
| `STRATA_Q4X_GRAPH_PDL` | on for a discrete CUDA card; off on an integrated GPU and on HIP | Qwen3.8's verify-window layer graphs (`STRATA_GLM_GRAPH`) are captured with PDL's programmatic edges where PDL is on (a capture the driver refuses is taken again without them). RTX 5090 + MTP: same tokens, same speed (190.1 / 190.0 tok/s). `0`: captured without the edges, as before v1.3.132 |
| `STRATA_Q4X_GDN_MID` / `STRATA_Q4X_GDN_MID_PF` | 1 / 0 (CUDA sm_90+; never on HIP) | Qwen3.8 decode: a delta-net layer's alpha / beta, conv + SiLU and recurrence in one launch after the qkv / z GEMV - the same arithmetic in the same order, the same bits (`gdn_core_parity`). `_PF=1`: it also pulls `ssm_out`'s weights into L2 during the recurrence (off: UD 3.1 122.7 tok/s without, 120.5 with - it competes with the state's reads). `STRATA_Q4X_GDN_MID=0`: the separate launches |
| `STRATA_GLM_ZERO_COPY` | on | the token boundary without the copy engines (v1.3.21): the table edits, the embedding row (a pinned copy of `token_embd`, Q8_0 / Q5_K) and the token are read and written host-mapped by the kernels. `0`: the copies, as before |
| `STRATA_GLM_GAP_PROF` | off | `1`: a `glm gap prof` line after each answer - where the time between two tokens goes (host laps, the GPU's idle gap by timing events) |
| `STRATA_GLM_TAIL_SKIP` / `_W` | off | a route's experts past rank r (or below a probability share) that are not in VRAM are left out - faster on a slow CPU lane, changes the answers (KL ~0.01-0.02) |
| `STRATA_GLM_ROUTE_AFFINITY` | off | `<a>`: a VRAM-resident expert's routing score gets +a (its weight stays the router's), so a near-tie goes to the expert already on the GPU (after Maya #252; v1.3.36). Changes the answers. GLM-5.3-Flash Maya-S-v2 on an RTX 5090: `0.04` -> 14.1 ms a token against 15.4 (-9%), VRAM hits 56 -> 69%, nll 1.631 against 1.635 (KL 0.043 to the default's logits); `0.08` -13% but nll starts to rise (1.641-1.651). Not on hash-routed layers |
| `STRATA_GLM_ROUTE_AFFINITY_RAM` | off | `<a>`: the same for experts in the RAM tier, a VRAM one getting the larger of the two (with an SSD tier: avoids experts that only the SSD has; v1.3.37 measured). DeepSeek V4.1 with the tail skip, 8 held-out sequences: `0.08` -8-12% a token (SSD reads -22%) for KL 0.2406 -> 0.2438 and ppl 9.76 -> 9.81; `0.16` -12-20% for KL 0.2501 / ppl 9.89 |
| `STRATA_GLM_ROUTE_PROF` | off | `1`: the route kernel prints its SM cycles per phase every 43,000 routes (diagnostics, NVIDIA) |
| `STRATA_GLM_TAIL_SKIP_DISK` | off | `1`: the tail skip leaves out only experts that are on the SSD alone (v1.3.29). With `STRATA_GLM_TAIL_SKIP=3` the DeepSeek V4.1 speed setting: 56.6 -> 34.2 ms a token, KL 0.229 -> 0.240; with `=4` (the 5th-6th) UD-IQ2_XXS on an AI PRO R9700 + lane went 48.5-53.4 -> 39.8 ms a token for KL +0.007 (`=3`: 38.6 ms, +0.020; 2026-10-27), the better trade there; changes nothing when every expert is in VRAM + RAM |
| `STRATA_GLM_RAM_FILL` | 2 with `STRATA_GLM_TAIL_SKIP_DISK=1`, else 0 | up to this many of the most routed experts that are on the SSD alone are read into the RAM tier in the background after each token (issue #8: the disk tail skip never computes the experts it leaves out, so they were never read and stayed on the SSD - 16-25 skips a token through a whole session). One is read over a colder RAM-tier expert when its route count beats that one's by `STRATA_GLM_RAM_FILL_MARGIN` (4). `0`: off. The engine log counts them (`glm fast tail skip ... RAM fill`) |
| `STRATA_GLM_DEFER` | off | expert deferral: the CPU lane's part of a layer's MoE is not waited for at the layer's end but written into the residual after the next layer's attention (with its own layer's weights) - the GPU never idles for the lane: single tokens 10.9 -> 9.5 ms, the 2-token verify window 14.4 -> 13.1, +12-15% tok/s on an RTX 5090 + Radeon 8060S; changes the answers more than the tail skip (KL 0.08 against the Q8_0 reference, 0.012 without; swe-bench L7 16 tasks x 3: 107.2 tok/s and resolved 13 of 16 against 100.9 and 14 with it off). With a layer split each part defers within its own layers (its last layer waits for its own part). AI PRO R9700 without the iGPU lane (v1.3.41, DeepSeek V4): 46.1-46.5 -> 53.0-53.2 tok/s one token at a time, KL 0.059 against the CUDA build's reference (0.0057 off; its top token 90% of the time, 97% off) |
| `STRATA_DS4_IDX`, `STRATA_DS4_ATTN` | 0 | `1`: the older FP32 kernels for the lightning indexer / the prompt attention instead of the tensor-core ones (A/B checks) |
| `STRATA_Q35_ATTN` | 0 | `1`: Ornith (qwen35moe) decode / verify-window attention on the older kernels (v1.3.64). 0: the split kernel (v1.3.65) - 8 keys summed in one shuffle tree, a softmax a 32-key tile, K / V through shared memory: 115K positions 1.09 -> 0.57 ms a layer a token, a 3-row window 3.29 -> 1.39. `STRATA_Q35_ATTN_PW` (query-head pairs a warp, 2) and `STRATA_Q35_ATTN_CHUNK` / `_CHUNK_MIN` (keys a block; default ~1/128 of the context, at least 64) are for tuning |
| `STRATA_Q35_PATTN` | 0 | `1`: Ornith (qwen35moe) prompts' attention on the older FP32 kernel (v1.3.66). 0: tensor cores (rocWMMA / CUDA WMMA, FP16 operands, FP32 sums; v1.3.67) - a 1,024-token sub-batch at 114K positions 918 -> 156 ms a layer; a 115K prompt 685 -> 154 s; KL 0.0054 against the FP32 kernel after a 24K context (two FP32 runs: 0.0064) |
| `STRATA_GLM_MMQ_GLU` | on | the prompt path's expert gate/up product writes the SwiGLU itself (NVIDIA Turing and newer); `0`: the separate SwiGLU kernel, bit-identical to older builds |
| `STRATA_GLM_CPU_MT` | on | CPU experts shared by several tokens (prompts, verify windows) decode each weight row once for all of them, bit-identical to ggml's dot (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS with the grid looked up in registers where `STRATA_GLM_CPU_1T` says, v1.3.41); `0` off |
| `STRATA_GLM_CPU_1T` | on | cheaper one-token rows on the CPU lane, bit-identical to ggml's dot: on an AVX-512 CPU with VNNI and VBMI (Zen 4 / Ice Lake and later) IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS with the grid looked up in registers (v1.3.41; a Ryzen AI Max+ 395's 16 threads, a DeepSeek V4 expert 109 -> 90 us, decode without the iGPU lane +3-9%; `STRATA_NO_IQ512_VBMI=1` only these off), else IQ3_S / IQ3_XXS in AVX2; `0` off |
| `STRATA_GLM_VRAM_EVICT` | counts | `lru`: the VRAM tier's refill evicts the least recently used expert (for cache studies) |
| `STRATA_GLM_DISK_READERS` | off | prompts read SSD-held experts with this many threads (try it on a DRAM-less NVMe; DeepSeek V4.1: `8`, a 1K-token prompt 33 -> 27 s) |
| `STRATA_GLM_DISK_PAR` | off | `1`: while answering, the experts a token misses are read from the SSD by a pool of their own with one thread for every chunk of an expert (3 parts x `STRATA_GLM_READ_CHUNKS` (8) = 24, the calling thread included), so an expert's chunks are all in flight at once instead of 9 threads taking them in about three rounds; `<n>` (2-64): that many threads. Only the order of the reads changes - the answers are the same. The threads wait inside the reads; between misses they spin `STRATA_GLM_DISK_PAR_SPIN_US` (50) µs and then sleep. With `STRATA_GLM_TIMING=1` the engine prints its disk reads (the rate while waiting, and the mean / p50 / p90 / p99 of the waits for one expert) either way, for an A/B. Not measured yet (fio on a 6.7 GB/s NVMe: 9 x 424 KB reads at once 4.64 GB/s, 24 at once 5.10 GB/s) |
| `STRATA_GLM_MIRROR` | off | `<folder>[:<folder>...]`: copies of the model's shards (same names) on another SSD; every expert read goes to whichever drive will finish first, for the reads while answering and the start-up fill. DeepSeek V4.1 UD-IQ2_XXS on an AI PRO R9700, drive 2 on USB4: SSD waits −20%, 32 GB + 64 GB +21% (v1.3.77, [SSD2](SSD2.md)). Linux |
| `STRATA_GLM_DISK_LANE` | off | `1`: an expert read from the SSD into RAM is computed there by the CPU / iGPU lane instead of being copied to VRAM (v1.3.29; -5% on V4.1 with Q2_K/Q3_K experts, nothing with the smaller Q2 ones) |
| `STRATA_DS41_ENGRAM_CACHE_MB` / `_THREADS` | 256 / 15 | DeepSeek V4.1: the Engram raw-row cache and the row readers (v1.3.29) |
| `STRATA_DS41_ENGRAM_ASYNC` | on | DeepSeek V4.1: a token's 48 Engram rows are read by a background thread while layer 0 runs, layers 1 / 14 wait for them on the GPU (v1.3.36; -0.5 ms a token, same logits); `0`: read before the token |
| `STRATA_DS41_ENGRAM_WINDOW_ASYNC` | on | DeepSeek V4.1: a verify window's Engram rows are read in the background the same way, each table's first layer waits for its rows on the GPU (v1.3.117; the window's Engram wait 0.39 -> 0 ms on an RTX 5090, same tokens); `0`: read before the window |
| `STRATA_DS41_ENGRAM_PREFETCH` | 8 | DeepSeek V4.1: threads reading a prompt chunk's Engram rows into the row cache right after its hash (v1.3.36; a 1K-token prompt's Engram 1.7-1.9 -> 0.6 s, the prompt -6-7%); `0` off |
| `STRATA_DS4_SPEC` | 1 (on) | speculative decoding from the prompt and the answer: draft tokens a step (1-3; 1 is the fastest measured); `0` off |
| `STRATA_DS4_SPEC_GATE` | on | drop a draft while the request's acceptance does not pay for the verify window; `0`: always draft |
| `STRATA_DS4_SPEC_MIN` | 3 | the tokens a draft's lookup matches (2-4) |
| `STRATA_DS4_SPEC_COST` | off | `1`: the window's size from what each width costs here and how often drafts are kept, instead of fixed rules (v1.3.117: the tables live as long as the process, a width's cost is the median of its last 16 steps, the single token is re-timed now and then). DeepSeek V4 0731 on an RTX 5090: same speed as off |
| `STRATA_DS4_NGRAM_LONG` | off | `1`: the n-gram drafter proposes from the longest earlier match and drafts more tokens the longer it is (n-gram-only decode; v1.3.117). Qwen3.8 copy task +8%, DeepSeek V4 0731 -12% (more windows): off by default |
| `STRATA_GLM_SAMPLE_COLS` | on | with temperature > 0, a verify window's columns are drawn in one launch and one wait instead of a copy, a launch and a wait a column (v1.3.117; same tokens - `STRATA_GLM_SAMPLE_COLS_CHECK=1` draws each column the old way too and counts differences); `0`: a column at a time |
| `STRATA_GLM_WINDOW` | on with unified memory (DGX Spark), off elsewhere | GLM-5.3-Flash on one GPU: speculative decode with verify windows - the model's own NextN block drafts up to `STRATA_Q35_MTP_K` (3) tokens and one batched window checks them (the dense weights read once for all columns), the same tokens as without (v1.3.118). DGX Spark: 21.4 -> 24.3 tok/s with the 94 GB S1 build, 19.8 -> 22.9 with unsloth UD-IQ2_XXS. A discrete card + RAM tier (RTX 5090) is break-even or slower, so it stays off there; `1` / `0` force it. Checks (stderr counts, all must be 0): `STRATA_GLM_WINDOW_CHECK=1/2`, `STRATA_GLM_REWIND_CHECK=1`, `STRATA_GLM_MVN_CHECK=1` |
| `STRATA_GLM_DENSE_Q8` | on | GLM-5.3-Flash: the GGUF's Q8_0 dense tensors (attention k_b / v_b, the indexer's, the KDA f/g/beta projections, the hc weights) are read as Q8_0 instead of a BF16 copy - 0.37 GB less read a token, KL 0.2708 -> 0.2667 (v1.3.118; DGX Spark +2%); `0`: the BF16 copy as before |
| `STRATA_GLM_KDA_SPLIT` | off | `1`: the KDA recurrence of one token in 4 column blocks a head (the same tokens; no gain measured on GB10 or the RTX 5090) |
| `STRATA_GLM_COLS_ORDER` | on (CUDA) | a verify window's column-batched expert launches (GLM-5.3-Flash, DeepSeek V4 / V4.1, the DSpark drafter) put the column first in the grid, so an expert two columns share is read from memory once and from L2 the second time (v1.3.119; the same blocks in another order, the same tokens). DGX Spark: GLM S1 and DeepSeek V4 0731 + DSpark both faster (see the changelog). `0`: the old launch order. The AMD build keeps its order |
| `STRATA_GLM_MLA_COLS` | on | GLM-5.3-Flash verify window: the absorbed attention k_b / v_b GEMVs of all columns in one launch each, each weight read once (v1.3.119; every column's sums as before, the same tokens). `0`: a column at a time |
| `STRATA_GLM_KDA_OOP` | on | GLM-5.3-Flash verify window: column 0 writes its KDA state straight into the rewind copy and the next column reads it from there, and a rewind replays from it the same way - no 2 x 146 MB of state copies a window (v1.3.119; the same sums). `0`: the copies |
| `STRATA_GLM_ROUTE_COLS` | on (Qwen3.8) | a verify window's routes, one per column, in one launch (v1.3.119: the same plans in the same order; a launch a column was 48 x T launches a window, ~13 us each on a DGX Spark). `1` on other models too, `0`: a launch a column |
| `STRATA_GLM_MVN_PDL` | on | the verify window's dense GEMVs (`mvn`) start with programmatic dependent launch and pull their weight rows into L2 before they wait for the launch before them, as the token path's GEMVs do (v1.3.119, CUDA; the same sums). `0`: launched as before |
| `STRATA_Q35_MTP_Q4` | off | `1` (experiment): Qwen3.8's MTP drafter reads 4-bit copies of its Q8_0 projections (IQ4_NL) and its head (Q4_K), made at load. The draft step is faster, fewer drafts are kept (DGX Spark: 87.6% -> ~80% accepted); the verify window decides every token, so the answer is the model's, but the tokens can differ from a run without it (v1.3.119) |
| `STRATA_DETERMINISTIC` | off | `1` (a test mode, slower): the choices that follow wall-clock timing are switched off - the prompt's CPU / iGPU share, the decode CPU / iGPU lane, the background tier moves and read-ahead, the n-gram gate, the pack's `expert_usage.txt` - and runs the prompt path's expert sums in one order whatever the pool holds (`STRATA_MMQ_WHOLE_K=1`, `STRATA_GLM_PREFILL_WINDOW=0`), so two runs of the same request give the same tokens (for A/B checks of a change that must not change them). A switch already set in the environment is kept. **A test switch, not a runtime setting:** it freezes the expert tiers where they start, so every expert outside them is fetched again for every token. On a PCIe-bound card that is a large loss - an RTX 4090 + 96 GB on Qwen3.8 UD-Q4_K_XL went from 54.1 to 14.6 tok/s (3.7x slower), RAM fetches ~20 -> 214 a token (#39). For the same answers run to run in normal use, Qwen3.8 has the exact lane combine (`STRATA_GLM_LANE_XC`, on by default since v1.3.127/v1.3.128) at no speed cost |
| `STRATA_DS4_SPEC_PIPE` | auto (HIP), 1 (NVIDIA) | the verify window: `1` pipelined (v1.3.43: the columns in two groups on the two hardware queues, batched inside a group), `0` batched (the dense weights read once for the window), `auto`: the first windows of each width time both (3 each) and the faster stays for that width (v1.3.40; per width from v1.3.43; the HIP default - not yet checked on an NVIDIA card). An RTX 5090 + iGPU: pipelined (swe-bench 90 against 73 tok/s); an AI PRO R9700 (two hardware queues, v1.3.43): pipelined at every width (2 / 4 / 6 / 8 columns 34-36 / 52-56 / 71 / 88 ms against 41 / 68.5 / 98 / 133 batched) |
| `STRATA_DS4_SPEC_GROUP` | half the window | the pipelined window's columns a group (v1.3.43); the default makes two groups of ceil(T / 2) |
| `STRATA_DS4_WIN_COLS` | on | a window group's VRAM-resident experts in one gate/up and one down launch for all its columns instead of a launch a column (v1.3.43; bit-identical: `STRATA_DS4_WIN_COLS_CHECK=1` compares every 16th group with a launch a column); `0`: a launch a column |
| `STRATA_GLM_CPU_MULTI` | on | a batched verify window's CPU-lane experts of a layer in one call for all its columns, a weight row decoded once for all of them (v1.3.43; bit-identical); `0`: a call a column |

`--max-context` sets the context. DeepSeek V4 is trained for 1M tokens; 65536 is the default and 131072 (a 125K
prompt) has been measured. The attention caches are small (0.44 GB at 65536), but the prompt path's buffers grow with
the context.

## Checking the output against llama.cpp

`tools/dsv4_logits_dump.cpp` writes llama.cpp's logits for a token sequence (build it against a llama.cpp build that
supports `deepseek4`); the engine's `STRATA_GLM_SCORE` mode compares its own logits with them. See
[docs/deepseek4/PORT.md](deepseek4/PORT.md) for the commands and the results: on UD-Q2_K_XL the two engines agree to
KL 0.009 over 3000 tokens, with the same next-token accuracy.

## Something went wrong?

- **The build stops at `mma with block scale not supported on .target sm_120`** - configure with
  `-DCMAKE_CUDA_ARCHITECTURES=120a`.
- **`the engine exited before it was ready`** - read the engine log (`log` in the config). Out of memory while pinning
  the RAM tier: lower `STRATA_GLM_RAM_GB` or raise `STRATA_GLM_RAM_HEADROOM_GB`.
- **`refers to missing files` / `pack:` errors** - the pack must sit next to the three shards it was made from; make it
  again after moving them.
- **The thinking stops after a word and the answer starts** - `max_tokens` is too small (see Thinking above).
- **Slow, the SSD busy** - not enough RAM + VRAM for the experts; close programs or add RAM.
- **Port in use** - another server runs on it; use `--port`.

## Limitations

- No MTP drafter: this GGUF has no MTP block. DeepSeek's DSpark drafter runs from v1.3.43 (`--dspark`, one GPU; see
  [The DSpark drafter](#the-dspark-drafter-deepseek-v4-flash)); without it speculative decoding drafts from the text
  itself (`STRATA_DS4_SPEC`).
- Conversation slots on disk (`STRATA_GLM_SLOTS`) are off for DeepSeek V4; the live reuse of the last conversation works.
- No images (DeepSeek V4 Flash is text-only here).
- Several GPUs: the layer split runs (see "Several GPUs"), and a slower second card
  does not beat one fast card with the iGPU lane - `auto` then runs on the fast card alone. Windows is untested for DeepSeek V4.
- `./abyss.sh` (Maya's installer) is for GLM; DeepSeek V4 is set up with `./setup-deepseek.sh`; DeepSeek V4.1 with
  `./setup-deepseek.sh --model v41` (zephel01's LDLQ GGUF) or by hand ([docs/deepseek41/README.md](deepseek41/README.md)).
- DeepSeek V4.1: no MTP drafter in any GGUF (the opt-in DSpark drafter, v1.3.53, drafts instead), no layer split, no
  conversation slots on disk. Up to v1.3.31 it crashed on the 16,385th token of a context (and chose its attention
  rows wrongly past 8K); use v1.3.32 or later for long chats; the prompt's attention is batched since v1.3.60.
