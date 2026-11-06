# Ornith-1.5-122B on Abyss

[Ornith-1.5-122B](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) is a Qwen3.5-MoE model (llama.cpp's
`qwen35moe`; MIT): 122B parameters, about 10B of them active a token. It is a different kind of model from
DeepSeek V4 and GLM-5.3-Flash, and the engine runs it through its own layer path (v1.3.46):

- **48 layers**: 36 **gated delta-net** layers (a linear-attention recurrence - 64 value heads of a 128 x 128 state
  each, a 4-tap causal conv in front, a sigmoid-gated SiLU norm behind) and, every fourth layer, 12 **full
  attention** layers (32 query heads of 256 over 2 KV heads, q / k RMS norms, RoPE on the first 64 dims with
  base 1e7, a sigmoid output gate), with a plain residual (no hyper-connections).
- **256 experts a layer, 8 routed** (softmax, renormalised) plus a **shared expert** behind its own sigmoid gate,
  all 1024 wide; the expert pool, the RAM tier and the CPU / iGPU lane work as for the other models.
- A NextN (MTP) block in the GGUF drafts the next tokens (below); the packer leaves it out of the pack, the engine
  reads it from the shards - or from a small repacked file, since unsloth's block keeps Q8_0 experts, which the
  decode's expert kernels do not take.

## Memory

| | |
| --- | ---: |
| experts (256 x 48, the pool / the RAM tier / the SSD) | 70.4 GB |
| dense weights, on the GPU (Q8_0: projections, the shared experts, the output head) | 5.4 GB |
| states: delta-net 149 MB, conv, KV caches (FP16: 24 KB a position) | ~0.2 GB + 1.6 GB at 64K |

A 32 GB card holds the dense weights and ~25 GB of experts (98 of 256 a layer); a 60 GB RAM tier
(`STRATA_GLM_RAM_GB=60` on a 94 GB box) holds the rest, so decode reads nothing from the SSD. With 64 GB of RAM part
of the experts stay on the SSD (`STRATA_GLM_TAIL_SKIP` applies as for DeepSeek V4).

## The GGUF

unsloth's `UD-Q4_K_XL` (79.7 GB, 3 shards; experts IQ4_XS / Q4_K / Q5_K / Q6_K by layer, everything dense Q8_0) is
the one measured here. unsloth's other sizes of the same repository have the same layout and should load (untested).

```sh
hf download unsloth/Ornith-1.5-122B-GGUF --include "UD-Q4_K_XL/*" --local-dir <models>/unsloth/Ornith-1.5-122B-GGUF
sha256sum <models>/unsloth/Ornith-1.5-122B-GGUF/UD-Q4_K_XL/*.gguf     # against the files' LFS checksums on the Hub
```

## The pack

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf <models>/unsloth/Ornith-1.5-122B-GGUF/UD-Q4_K_XL/Ornith-1.5-122B-UD-Q4_K_XL-00001-of-00003.gguf \
  --out <models>/unsloth/Ornith-1.5-122B-GGUF/UD-Q4_K_XL/pack
```

The packer recognises `qwen35moe`, indexes the experts in place (nothing is copied), keeps every dense tensor as the
GGUF stores it (the engine's quantized GEMV reads Q8_0 directly) and writes the tokenizer and the chat template; the
NextN block (`blk.48`) is left out. The pack directory is ~160 MB.

## Running it

`./setup-deepseek.sh --model ornith --download-model` installs zephel01's IQ3_XXS-B (16 GB GPU + 64 GB RAM) with the
engine, the pack and the config. For unsloth's files, build the engine (the manual's [Build](../MANUAL.md#build)) and
write a config next to the others:

```json
{
 "exe": "<Abyss>/build/strata",
 "args": ["--glm-pack", "<dir>/UD-Q4_K_XL/pack", "--max-context", "65536", "--prefill", "auto"],
 "cwd": "<Abyss>",
 "tokenizer": "<dir>/UD-Q4_K_XL/pack/tokenizer",
 "model_name": "ornith-1.5-122b",
 "gpu": [0],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
 "port": 8082,
 "log": "<Abyss>/abyss-ornith.log",
 "env": {"STRATA_GLM_RAM_GB": "60", "STRATA_GLM_USAGE": "0"}
}
```

(an AMD card: `"exe": "<Abyss>/build-hip/strata"`, `"backend": "hip"`, `"lib_dirs": ["/opt/rocm/lib"]` and the
`GPU_MAX_HW_QUEUES` / `HIP_VISIBLE_DEVICES` entries of [docs/amd](../amd/README.md)). The sampling defaults are the
model card's. Then:

```sh
systemd-run --user --scope -p MemoryMax=88G .venv/bin/python serve/server.py --engine strata \
  --config abyss-ornith.json --port 8082
```

## Speed and quality (AI PRO R9700 + Radeon 8060S lane + 94 GB RAM, ROCm 10)

| | |
| --- | ---: |
| decode (every expert in VRAM or the RAM tier, the 8060S computing the lane) | 23.0-24.6 ms a token (41-43 tok/s) |
| ... the CPU computing the lane instead (`STRATA_GLM_IGPU=0`: a PC without an iGPU) | 30.3 ms a token (33 tok/s) |
| a copy task with the engine's n-gram drafts (the default, `STRATA_DS4_SPEC=1`; `=3`: three drafts) | 17.4 ms a token (57 tok/s); 13.1 (76 tok/s) |
| with the NextN block's drafts (`STRATA_GLM_MTP_GGUF`, below): code / prose / the copy task | 16.9 (59 tok/s) / 20.0 (50) / 11.8 (85) |
| ... the CPU computing the lane (no iGPU): code / prose | 24.0 (42 tok/s) / 25.5 (39), from 28.0 / 30.3 |
| a 575-token prompt through the batched path (chunks up to 32768, sub-batches of 1024) | 359 tok/s |
| decode after a 57.6K / 115K-token prompt (IQ3_XXS-B, 16 GB of VRAM, no lane, NextN; v1.3.65) | 37.9 / 34.0 tok/s, from 30.5-31.5 / 23.7-24.6 |
| a repeated ~5.5K-token prompt / its next turn (conversation reuse, v1.3.65) | 8.1 -> 1.4 s / 8.0 -> 1.8 s |
| a 57.6K / 115K-token prompt (the same setup; the attention on tensor cores, v1.3.67) | 1,076 / 747 tok/s (54 / 154 s), from 324 / 168 |
| an 8K-token prompt (UD-Q4_K_XL, `--bench`, v1.3.67) | 1,348 tok/s, from 948 |
| KL against llama.cpp (its CPU backend), 559 positions of a 576-token text | 0.024 |
| ... the same text, the first half through the prompt path, the 287 positions after it | 0.026 |

For scale: on this model llama.cpp's own CPU and HIP backends differ by KL 0.027-0.029 over the same positions (the
quantisation noise of 48 layers), so both paths sit at that floor. Where a decode token goes (the R9700's stream,
`STRATA_GLM_PROF=1`): the delta-net projections 5.0 ms, the recurrence and its norms 3.6 ms, the lane's answer 2.5,
the pool's experts 3.4, `ssm_out` 2.0, the attention projections 1.4, the shared expert and router 1.8, the norms 1.0.
Without the iGPU the CPU lane (16 threads, 0.046 ms an expert, ~200 experts a token) is what the GPU waits for:
7.2 ms a token. The prompt: the staged experts 63% (the RAM tier's, computed on the lane or copied), the delta-net
projections 19%; the router runs on the engine's own tile kernel on HIP (rocBLAS's F32 GEMM took 15.7 ms a layer
inside the engine, a third of the prompt, against 0.2 ms for the same shape alone).

**Speculative decoding.** The engine's n-gram drafts (the latest earlier occurrence of the last tokens in the prompt
and the answer proposes the next ones, [the manual](../MANUAL.md#memory-and-speed)) run on this model through its
verify window: a token and its drafts go through every layer at once (the dense weights read once for the window's
columns, the recurrence and the conv carried through the columns in order, the attention over the caches), each
column's sample is compared with the proposal, and a rejected tail is dropped - the delta-net and conv states are
replayed from the kept columns. The output is the token-at-a-time loop's (the same sampler on the same logits; the
greedy tokens matched over 160-token answers of three texts). It pays on repeated text (copies, code with repeats:
97% of 3 drafts accepted, 76 tok/s) and costs little elsewhere (one draft a window by default; on code with few
repeats 45 tok/s against 43).

**The NextN block's drafts.** The GGUF's draft block (`blk.48`: an attention layer and a MoE behind `eh_proj`, fed a
position's normed hidden state and the next token's embedding) proposes the tokens after the one just sampled, up
to 3 chained (`STRATA_Q35_MTP_K`), each with its softmax probability; the decode keeps the drafts above
`STRATA_DS4_DSPARK_THR` (0.5 for this drafter) and verifies them in one window. unsloth's block keeps its experts at Q8_0, which
the decode's expert kernels do not serve, so the engine says so and drafts nothing - repack the block once:

```sh
.venv/bin/python tools/nextn_pack.py --gguf <dir>/UD-Q4_K_XL/Ornith-1.5-122B-UD-Q4_K_XL-00001-of-00003.gguf \
  --out <dir>/UD-Q4_K_XL/Ornith-1.5-122B-nextn-q4k.gguf --ggml-lib <llama.cpp build>/bin/libggml-base.so
```

(1.5 GB, two minutes; the experts go to Q4_K through ggml's quantizer - any llama.cpp build's `libggml-base`), and
put `"STRATA_GLM_MTP_GGUF": "<dir>/UD-Q4_K_XL/Ornith-1.5-122B-nextn-q4k.gguf"` in the config's `env`. The block's
experts take their share of the expert pool and the RAM tier like a 49th layer (its cache rows are written for every
position, prompt and answer: eh_proj and k / v alone). Measured, greedy, 160-token answers: code 92% of the drafts
accepted, 22.3 -> 16.9 ms a token (59 tok/s); prose 68-74% accepted, 21.6 -> 20.0-20.8 (48-50 tok/s); the copy task 85 tok/s.
Without the iGPU the gain is larger (the lane's latency is paid once a window): code 35.7 -> 41.7 tok/s, prose
33 -> 39. `STRATA_GLM_NO_MTP=1` leaves the block out.

NVIDIA: on an RTX 5090, zephel01's IQ3_XXS-B decodes at 116 tok/s with its NextN drafter (75% of the drafts accepted;
v1.3.100, from the config `./setup-deepseek.sh --model ornith` writes). Given issue #23's small pool (1.45 GB of
experts, an 8 GB card's) it read prompts at 356 tok/s (2K) / 347 (8K) and decoded at 45.5 tok/s, KL 0.0073 against a
16 GB run (v1.3.82).

## Not yet

- Conversation slots on disk (`STRATA_GLM_SLOTS`; the live reuse of the last conversation works from v1.3.65); a
  layer split; images.
- The lane's expert kernels are tuned for DeepSeek's and GLM's shapes - Ornith's 3072 x 1024 experts take the
  generic ones. The recurrence kernel runs far below the memory bound. Both are the next speed-ups.

## Checking it

`GLM_CB_DIR=<dir>` writes the token path's seams under llama.cpp's cb names (`attn_norm-<layer>.f32`,
`conv_output_silu-<layer>`, `final_output-<layer>`, `attn_gated-<layer>`, `ffn_out-<layer>`, `l_out-<layer>`, ...;
`GLM_CB_POS=1` keeps a file a position), `GLM_PF_CB_DIR` the prompt path's last row, and `STRATA_GLM_SCORE=<c>` with
`STRATA_GLM_SCORE_PREFILL=1` scores a text after a prefilled prefix - the manual's
[Checking the output against llama.cpp](../MANUAL.md#checking-the-output-against-llamacpp) has the reference side.
