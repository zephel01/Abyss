# Small VRAM: 8-12 GB cards

[日本語](SMALL-VRAM.ja.md)

Abyss is tested on 16 GB + 64 GB RAM (and 32 GB cards). An 8 GB laptop GPU runs the same models, with less room: the
dense weights (attention, shared experts, embeddings) take 5-6 GB of it, the KV cache some more, and the expert pool
gets what is left - a few slots a layer. This page says what Abyss does with that little (from v1.3.82, issue #23),
what the refusal and warning lines mean, and the one change that helps most on GLM-5.3-Flash: a lighter dense pack
(IQ4_XS), made with one command.

Measured so far (issue #23, an RTX A2000 Laptop 8 GB, PCIe Gen4 x8, Ryzen 9 7940HX, 64 GB DDR5, Abyss v1.3.77):
Ornith-1.5-122B IQ3_XXS-B decodes at 15.8 tokens/s and read its prompt at 16 tokens/s (token by token). GLM-5.3-Flash
Maya-S-v2 did not start until its dense pack was re-quantized to IQ4_XS, then 4-4.7 tokens/s with a 512 GB SSD
(disk-bound: the RAM tier holds 40 GB of its 96).

Measured with v1.3.82 on an RTX 5090 given the reporter's pool (`STRATA_GLM_POOL_GB`, see the last section; PCIe Gen4
x4, 94 GB RAM), `./abyss.sh --bench`:

| | v1.3.81 behaviour | v1.3.82 |
|---|---|---|
| Ornith-1.5-122B IQ3_XXS-B, pool 1.45 GB (8 slots a layer) - prompt 2K / 8K | 46 tok/s (token by token) | **356 / 347 tok/s** (chunks of 768) |
| the same - decode | 48.3 tok/s | 45.5 tok/s (run-to-run spread 42-49) |
| the same - quality (KL vs a 16 GB run's reference) | 0.0073 | 0.0073 |
| GLM-5.3-Flash Maya-S-v2 (Q6_K dense), pool 1.2 GB (4 slots a layer) | refused (`not enough VRAM`) | **starts**, decode 9.1 tok/s (SSD-bound, RAM tier 40 GB), KL 0.039 (a 32 GB run: 0.036-0.044) |

A laptop card has a smaller x8 link and CPU, so expect less than these on the same pool; the ratio of the prompt
numbers (x7.7) is the point.

## Contents

- [What Abyss does on a small card](#what-abyss-does-on-a-small-card)
- [The lines in the log](#the-lines-in-the-log)
- [GLM-5.3-Flash: the IQ4_XS dense pack](#glm-53-flash-the-iq4_xs-dense-pack)
- [Ornith-1.5-122B on 8 GB](#ornith-15-122b-on-8-gb)
- [Settings](#settings)
- [Trying it on a bigger card](#trying-it-on-a-bigger-card)

## What Abyss does on a small card

Three things changed in v1.3.82 for a pool of a few slots a layer:

1. **The pool may be smaller than a route.** GLM-5.3-Flash and Ornith route 8 experts a token through every layer.
   Before, the pool had to hold at least 8 slots a layer or the engine refused to start ("not enough VRAM for the
   expert pool"). Now it starts with 4 or more: an expert that is not in VRAM lands in a spare slot or, with none
   free, in a scratch slot of its own, and the CPU lane computes its share from RAM - none of that needs the route to
   fit. The reporter of issue #23 ran 6 of 8 that way, bit for bit the same output as the engine's own reference
   (KL 0). Such a pool misses most of every route, so it is slow; the dense pack below is the better fix. Below 4 the
   engine still refuses, and says how much VRAM is short (next section).
2. **The prompt is read in chunks, not token by token.** The prompt path borrows its buffers from the expert pool, and
   used to leave every layer a route's experts, the spares and 2 (13 slots): a pool of 9 could lend nothing, and the
   prompt ran one token at a time - 16 tokens/s in issue #23. Now a pool that can lend no chunk at that keeps only 4
   slots a layer while a prompt runs and lends the rest (a pool of 9 lends 5 - on Ornith ~800 MB, chunks of ~2,000
   tokens). A prompt runs alone and stages nearly every expert of a layer whatever the pool holds, so the lent
   residents cost it little, and the decode's misses refill the tail after it (1.5 GB, a fraction of a second).
   Estimate for the 2,048-token prompt of issue #23: the experts stream once per chunk per layer (42 GB over PCIe
   Gen4 x8, ~3 s) plus the GPU's own matrix work - about 150-400 tokens/s against 16.
   Since v1.3.137 the small keep is also taken when the full keep lends only a short chunk (under 4,096 tokens) and
   the small keep lends one at least twice as long (`STRATA_GLM_PREFILL_SMALL_BELOW`), and the chunk is sized again
   against the pool the engine really carved when that came out smaller than the first estimate (a drafter or the
   context took VRAM in between). Ornith-1.5-397B 16G64G on a 16 GB card: 19 slots a layer lent 4 (chunks of 1,024,
   4K prompts ~70 tokens/s) and now lend 15 (chunks of 12,288, ~210 tokens/s; 8K prompts 65 -> 390); a pool of 15
   slots that lent nothing (a 4K prompt ~400 s token by token) now prompts at ~250 tokens/s.
3. **The CPU lane takes every RAM-tier expert when it is much faster than the link.** The split between the CPU lane
   and PCIe is measured at start (`an expert 0.055 ms on the CPU vs 0.282 ms over PCIe` in issue #23) and used to keep
   one expert of 8 on the link. On that box every expert on the CPU was 10% faster (the `--deep` sweep: share 0 at
   18.5 tokens/s against the engine's 16.8): both lanes read the same DRAM, and the link copy sits on the GPU's
   stream before the layer's experts. When the CPU does an expert 4x faster than the link (`STRATA_GLM_LANE_LEAN`),
   every RAM-tier expert now goes to the CPU - once the pool has settled (the hit rate stops rising), the measured
   split until then so a prompt's misses still refill the pool (issue #14), and back when the hit rate falls. Cards
   whose split already sends every expert to the CPU (a Gen4 x4 or OCuLink link) are as before, and so is a card
   whose link is close to the CPU (an R9700 on Gen5 x16: 0.311 vs 0.499 ms).

Nothing changes on a card whose pool holds a route and lends a chunk: the three rules only apply below those points.

## The lines in the log

```
glm fast: CUDA0 VRAM before the expert pool: 5.72 of 7.71 GB in use - dense weights 5.17, state/KV (8192 ctx) 0.00, ...
glm fast: CUDA0 layers [0,48) expert pool 1.45 GB, 9-9 slots/layer (432 total)
```

Where the VRAM goes: the dense weights are the pack's attention / shared-expert / embedding tensors (the IQ4_XS pack
below makes them ~2 GB smaller on GLM-5.3-Flash), state/KV grows with `--max-context` (`--kv q4_0` halves it), and the
pool gets the rest less a reserve (`STRATA_GLM_RESERVE_MB`, 700 MB on a CUDA card of 12 GB or less).

```
glm fast: not enough VRAM for the expert pool: 3 slots per layer, 4 at least (170 MB short; a whole route's 8 slots:
  850 MB short) - a smaller --max-context, --kv q4_0 or a lighter dense pack (tools/requant_dense.py IQ4_XS,
  docs/SMALL-VRAM.md) frees it
```

The refusal: the first number is what is missing for the engine to start at all, the second what would hold a whole
route (the pool stops missing most of every token from there). Any one of the three changes frees VRAM:
`--max-context 8192` instead of 65536 (the KV cache), `--kv q4_0` (the KV cache at a quarter of fp16), or the dense
pack (the biggest, ~2 GB on GLM-5.3-Flash).

```
glm fast: CUDA0 the expert pool holds 6 slots a layer, below a route's 8: the rest of every route runs from the
  scratch slots and the CPU lane (340 MB more VRAM would hold a route; docs/SMALL-VRAM.md)
glm prefill: CUDA0 the expert pool (9 slots a layer) can lend no prompt chunk while every layer keeps 13 slots - it
  keeps 4 while a prompt runs (small VRAM; STRATA_GLM_PREFILL_KEEP)
glm prefill: CUDA0 chunks of 2048 tokens (--prefill auto, up to 32768; ...)
glm fast: CUDA0 CPU lane: the CPU does an expert 5.1x faster than the link - every RAM-tier expert goes to the CPU
  once the pool settles (windows of 128 token steps), the split below until then and when the VRAM hits fall
  (STRATA_GLM_LANE_LEAN=0 keeps the split)
```

The three rules above, as they report themselves. A prompt path that still finds no chunk says what the smallest one
needs and what the pool can lend:

```
glm prefill: CUDA0 the expert pool (4 slots a layer, 4 kept) can lend no prompt chunk (--prefill auto): a 256-token
  chunk needs 120 MB, the pool can lend 0 MB - 120 MB of VRAM short; a smaller --max-context, --kv q4_0 or a lighter
  dense pack (docs/SMALL-VRAM.md) frees it - token by token
```

## GLM-5.3-Flash: the IQ4_XS dense pack

Maya-S-v2's dense tensors are Q6_K: 5.6 GiB in the GGUF, 5.97 GB of VRAM with the engine's own tables. Re-quantized
to IQ4_XS they are 3.6 GiB (the reporter of issue #23 measured 4.68 -> 3.61 GiB from a Q5_K version, and the VRAM
line `dense weights 5.97 -> 4.92`), which turns a pool of 4 slots a layer into one of 8: a whole route. The
quality cost of IQ4_XS on these tensors is not yet measured against the Q6_K pack (the reporter's run was
deterministic against itself, KL 0); the Q5_K attention pack on DeepSeek V4 cost KL 0.006 -> 0.012
([the manual](MANUAL.md#a-lighter-dense-pack-the-attention-projections-at-q6_k-or-q5_k)).

Which tensors: the attention projections (`attn_output`, `attn_q`, `attn_k`, `attn_v`, `attn_q_a`, `attn_q_b`,
`attn_kv_a_mqa`), the shared experts (`ffn_*_shexp`) and the three dense FFN layers 0-2 (`ffn_down/gate/up`).
Kept as they are: `token_embd` and `output` (Q6_K), `attn_k_b` / `attn_v_b` (Q6_K, 3-D tensors the re-quantizer
must not touch), the F32 norms and router, and the Q8_0 recurrence / indexer tensors. The routed experts stay in
the GGUF files untouched.

`tools/requant_dense.py` needs a llama.cpp build's `libggml-base.so` (the quantizer; `llama-quantize` itself refuses
an i-quant model without an imatrix) and gguf-py (Abyss's `third_party/llama.cpp/gguf-py`). It rewrites the 96 GB
of shards (~15 minutes on an NVMe) into a new directory beside the original, then `iq_pack.py` builds the pack:

```sh
cd ~/llm/apps/Abyss
M=~/llm/apps/Maya-data/models
GGUF_PY=$PWD/third_party/llama.cpp/gguf-py LIBGGML=~/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so \
  .venv/bin/python -I tools/requant_dense.py $M/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  $M/Maya-S-v2-IQ2_XXS-iq4dense iq4_xs --from=q6_k \
  'blk\.\d+\.attn_output\.weight' 'blk\.\d+\.attn_[qkv]\.weight' 'blk\.\d+\.attn_q_[ab]\.weight' \
  'blk\.\d+\.attn_kv_a_mqa\.weight' 'blk\.\d+\.ffn_(down|gate|up)_shexp\.weight' 'blk\.[0-2]\.ffn_(down|gate|up)\.weight'
.venv/bin/python tools/iq_pack.py --gguf $M/Maya-S-v2-IQ2_XXS-iq4dense/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  --out $M/Maya-S-v2-IQ2_XXS-iq4dense/pack --compat-bf16
cp -r $M/Maya-S-v2-IQ2_XXS/pack/tokenizer $M/Maya-S-v2-IQ2_XXS-iq4dense/pack/
```

(`abyss.py --repack` only takes the model names it knows, so `iq_pack.py` is called directly. `--from=q6_k` names the
source type: the script re-quantizes Q8_0 tensors only unless told otherwise.) Stop the server while it runs. The new
directory needs 96 GB; the original can go once the new pack works (the reporter deleted the intermediate to make
room).

Then the config (`abyss-glm.json` or the one the installer wrote) points at the new pack, with the context and KV
cache sized for the card:

```json
  "args": ["--glm-pack", "/home/you/llm/apps/Maya-data/models/Maya-S-v2-IQ2_XXS-iq4dense/pack",
           "--max-context", "8192", "--kv", "q4_0", "--prefill", "auto"],
  "tokenizer": "/home/you/llm/apps/Maya-data/models/Maya-S-v2-IQ2_XXS-iq4dense/pack/tokenizer",
```

`./abyss.sh --bench` then measures it (the quality check against `bench/ref` says whether the IQ4_XS pack writes the
same tokens as the Q6_K one). Quality: `.venv/bin/python tools/bench_ref.py <config> --tokens 512 --k 32 --check`
scores the pack against the engine's own reference (nll, top-1).

On a 512 GB SSD the model's 96 GB sit beside the system, and the RAM tier (~40 GB of 64) holds under half of the
experts: the SSD reads bound the speed (44 reads a token, 176 ms, in issue #23). A faster SSD or a second one
([SSD2](SSD2.md)) helps there more than the GPU does.

## Ornith-1.5-122B on 8 GB

[zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF](https://huggingface.co/zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF)
(51.7 GB) runs on 8 GB + 64 GB without any re-quantizing: its dense weights take 5.17 GB, the pool 9 slots a layer,
and the whole model fits the RAM tier (no SSD reads once warm). Issue #23's numbers on an RTX A2000 Laptop 8 GB:
decode 15.8 tokens/s, prompt 16 tokens/s with v1.3.77. With v1.3.82 the prompt runs in chunks (x7.7 on the same
pool, the table at the top) and the CPU lane takes
every RAM-tier expert once the pool settles (the share sweep's 18.5 tokens/s), so `STRATA_GLM_PCIE_SHARE=0` need not
be set by hand. The config: `--max-context 8192` (the KV cache is small on this model; 16K fits too).

## Settings

| setting | default | what it does |
|---|---|---|
| `STRATA_GLM_POOL_FLOOR` | 4 | the expert pool's smallest size in slots a layer (the spares and one resident); below it the engine refuses to start and says how much VRAM is short. A route's experts (8) were the floor before v1.3.82 |
| `STRATA_GLM_PREFILL_KEEP` | 4 | slots a layer the pool keeps while a prompt runs on a pool that could lend no chunk at the full keep (a route, the spares and 2). Larger keeps more experts resident through the prompt and gives the prompt a smaller chunk |
| `STRATA_GLM_PREFILL_SMALL_BELOW` | 4096 | the small keep is also used when the full keep lends a chunk shorter than this many tokens and the small keep lends one at least twice as long. `0`: the small keep only when the full keep lends no chunk (v1.3.136) |
| `STRATA_GLM_LANE_LEAN` | 4 | when the CPU lane does an expert this many times faster than the PCIe link, every RAM-tier expert goes to the CPU once the pool has settled. `0`: the measured split only, as before v1.3.82. `STRATA_GLM_PCIE_SHARE` / `_LATE` / `STRATA_GLM_CPU_PLAN` set by hand win |
| `STRATA_GLM_VRAM_GB` | off | behave like a card with this many GB: the expert pool gets what such a card would have (the context, dense weights and state already held count). For trying this page's rules on a bigger card |
| `STRATA_GLM_RESERVE_MB` | 700 (CUDA, 12 GB or less), 1536 (bigger CUDA cards, HIP) | VRAM left free beside the pool; issue #23 ran with 64 on CUDA, which gained ~0.6 GB of pool |

## Trying it on a bigger card

`STRATA_GLM_VRAM_GB=8` in the config's `"env"` makes a 16 or 32 GB card size its pool as an 8 GB one would (the
prompt path sizes its chunk from the same cap). The dense weights and KV cache are allocated as usual, so the cap must
be at least what they take plus the reserve. The log's `VRAM cap` line says what was left for the experts:

```
glm fast: CUDA0 VRAM cap 8.0 GB: 5.72 GB already in use, 1.60 GB left for experts
```

A big desktop card's CUDA context, MTP head and scratch take more than a laptop's (1.41 GB on an RTX 5090 with
Ornith, against the ~1 GB the 8 GB reports imply), so under `STRATA_GLM_VRAM_GB=8` it gets a smaller pool than the
real 8 GB card - 4 slots a layer on Ornith instead of 9, and a pool at the floor lends no prompt chunk. To repeat a
report, give the pool its size instead: `STRATA_GLM_POOL_GB` = the `expert pool N GB` of the reporter's log (the
table at the top used 1.45 for Ornith, 1.2 for GLM-5.3-Flash).
