# Laguna-S-2.1 on Abyss

[Laguna-S-2.1](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF) (poolside; llama.cpp's `laguna`) is a MoE model of
118B parameters, about 8B of them active a token. The engine runs it on the Qwen3.5-MoE layer path with its own
attention:

- **48 attention layers**: every fourth (12) full attention, the rest (36) a **512-token sliding window**. 8 KV heads
  of 128; 48 query heads on the full layers, 72 on the window layers. q / k RMS norms, NEOX RoPE (full layers: the
  first 64 dims, base 5e5 with **YaRN** x32 over an 8K original context; window layers: all 128 dims, base 1e4), and
  a per-head **softplus output gate** from its own projection.
- Layer 0 a dense SwiGLU FFN; layers 1-47 **256 experts, 10 routed** (sigmoid scores, a selection bias, the weights
  renormalised and x2.5) plus an ungated **shared expert**; the expert pool, the RAM tier and the CPU lane work as
  for the other models. No NextN block.
- The window layers keep a ring of 2,560 rows (the window + a prompt slice) instead of the whole context: the KV
  cache is ~0.7 GB at 8K and grows only by the 12 full layers (24 KB a position).

## The GGUF and the pack

unsloth's `UD-IQ4_XS` (3 shards, 58 GB: experts IQ3_S / IQ4_XS, a few Q6_K; dense Q8_0) is the one measured here.

```sh
./setup-deepseek.sh --model laguna --download-model
```

downloads it, builds the engine, packs it and writes `abyss-laguna-iq4_xs.json` (the RAM tier the PC's RAM allows;
the model card's sampling: temperature 1.0, top_k 20). By hand:

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf <models>/unsloth/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf \
  --out <models>/unsloth/Laguna-S-2.1-GGUF/UD-IQ4_XS/pack
```

The chat template thinks by default (`<assistant><think>`); `"chat_template_kwargs": {"enable_thinking": false}`
turns it off. The server stops at `</assistant>`.

## Speed and quality (RTX 5090 capped, no iGPU, a 50 GB RAM tier)

A 4,058-token prompt and 512 greedy tokens, 32K context, three fresh runs, the median; llama.cpp on the same PC with
the same caps for comparison:

| VRAM | decode | prompt | llama.cpp decode / prompt |
| --- | ---: | ---: | ---: |
| 16 GB | 42.5 tok/s | 859 tok/s | 32.0 / 318 |
| 24 GB | 50.9 tok/s | 1,005 tok/s | 38.1 / 391 |
| 32 GB | 60.6 tok/s | 1,174 tok/s | 43.2 / 470 |

(v1.3.142: the n-gram drafts and a faster decode attention; v1.3.140 had 36.3 / 41.4 / 47.4 tok/s decode and 842 / 975 /
1,144 prompts. The drafts help on edit tasks - +5-14% on a 400-token code rewrite at 24 GB - and little on plain prose.
The verify window costs ~0.4 GB of VRAM. Outputs are bit-identical to v1.3.141's.)

Every expert sits in VRAM or the RAM tier (nothing from the SSD); the decode waits on the CPU lane (the RAM tier's
experts, 10-17 ms a token).

KL against llama.cpp (CUDA) over 3,071 tokens of English, Japanese and code: 0.024 (top token 93.4%); the prompt
path 0.019; positions 8,192-10,239 of a long text (past YaRN's original context) 0.014. llama.cpp's own CPU backend
differs from its CUDA one by 0.024 over the same tokens: the router's top-10 of 256 has many near-ties, so both
engines sit at that floor.

## Not yet

- A NextN drafter (no NextN block in the GGUF; the n-gram drafts work from v1.3.142).
- AMD: the HIP build compiles (gfx1201 / gfx1151) but has not been run.
