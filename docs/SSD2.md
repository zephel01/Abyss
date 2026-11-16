# Two SSDs (STRATA_GLM_MIRROR)

[日本語](SSD2.ja.md)

Experts that do not fit in VRAM and RAM are read from the SSD while Abyss answers. On DeepSeek V4.1 that SSD wait is
30-40% of a token's time. Copy the model's files to a second SSD too, and Abyss reads from both drives, cutting the
wait.

A PC whose memory slots are full can often still take another NVMe in a free M.2 slot. This feature is for those PCs.
An NVMe left over from an upgrade works too: put it in an M.2 slot or a USB4 / Thunderbolt enclosure and copy the
files. Drive 2 may be slower than drive 1 (each drive takes reads by its speed); a SATA SSD (~0.5 GB/s), though, takes
almost none.

## Contents

- [How much it helps](#how-much-it-helps)
- [How to use it](#how-to-use-it)
- [Space on the second drive](#space-on-the-second-drive)
- [How it works](#how-it-works)
- [Where it helps and where it does not](#where-it-helps-and-where-it-does-not)
- [Against RAID 0](#against-raid-0)
- [Other engines](#other-engines)
- [Limits](#limits)

## How much it helps

DeepSeek V4.1 Flash 1017, unsloth UD-IQ2_XXS (the 2026-10-28 files), AI PRO R9700 (OCuLink Gen4 x4) + the iGPU lane,
no tail skip. Medians of 2 sequences x 3 rounds, one and two drives alternating (2026-10-29).

- Drive 1 (where the model lives): Lexar NM790 4TB, internal M.2, 6.2 GB/s
- Drive 2: an NVMe in a USB4 enclosure, 3.5 GB/s

| GPU + RAM | A token (1 → 2 drives) | Speed | SSD wait a token | Read from drive 2 | Start-up fill |
| --- | --- | --- | --- | --- | --- |
| 32 GB + 96 GB | 51.4 → 49.2 ms | +4.5% | 19.7 → 15.7 ms (−20%) | 37-45% | 5.2 → 6.9 GB/s |
| 32 GB + 64 GB | 71.9 → 59.3 ms | **+21%** | 27.9 → 22.2 ms (−20%) | 38-45% | 4.4 → 6.5 GB/s |
| 16 GB + 64 GB | 92.0 → 77.7 ms | **+18%** | 42.8 → 34.0 ms (−21%) | 39-47% | 5.8 → 7.6 GB/s |

- The output does not change (the same file is read).
- The larger the SSD wait, the more it helps: the less RAM, the bigger the gain.
- Drive 2 is a USB4 enclosure at about half drive 1's speed, so it takes about 40% of the reads. Two M.2 drives of
  similar speed split them about in half and cut the wait further - see the next table.

**Two M.2 drives of similar speed** (reported in [#48](https://github.com/zephel01/Abyss/issues/48), 2026-11-16): the same
UD-IQ2_XXS, `--max-context 507904`, tail skip 4 / disk 1, `RAM_GB=44`, v1.3.154, HIP. RX 9070 16 GB (Gen5 x16), Ryzen 7
7800X3D, 64 GB. Drive 1: Crucial T500 2 TB (PCIe 4.0, CPU-attached M.2, also the OS disk). Drive 2: Samsung 990 PRO 1 TB
(PCIe 4.0, **chipset** M.2, a 400 GB partition), shards 1-5.

| GPU + RAM | A token (1 → 2 drives) | Speed | SSD wait a token | Read from drive 2 | Start-up fill |
| --- | --- | --- | --- | --- | --- |
| 16 GB + 64 GB | 81.3 → 65.1 ms | **+25%** | 38.9 → 22.3 ms (**−43%**) | ~45% (53.5% with the Q5_K dense overlay) | 4.5 → 9.7 GB/s |

- Prompts 2K / 8K: 59 / 79 → 108 / 142 tok/s.
- Earlier on the same PC (v1.3.129, 32K): V4.1 UD-IQ1_M SSD wait 36.8 → 20.1 ms (−45%), decode 13.5 → 17.0 tok/s;
  GLM-5.3-Flash 32G96G 73.0 → 39.9 ms (−45%), 7.7 → 11.2 tok/s.
- A chipset M.2 slot was enough for drive 2 here (nothing else busy on the chipset), even as a partition of a disk
  shared with another OS.
- An overlay pack (`tools/dense_overlay.py`) opens the shards through links named `overlay-NNNNN-of-NNNNN.gguf`; from
  v1.3.160 the mirror is also found by the original shards' names, so one copy on drive 2 serves both packs.
- The drops when the SSD has a slow spell (they happen now and then) got smaller too.

## How to use it

1. Copy the model's files, **with the same names**, to the second SSD. Not all of them are needed: only the shards you
   copy are read from both drives.

   ```bash
   # e.g. DeepSeek V4.1 UD-IQ2_XXS to /mnt/ssd2 (shard 6, the Engram table only, is not needed)
   mkdir -p /mnt/ssd2/V4.1-UD-IQ2_XXS
   cp ~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS/*-0000[1-5]-of-00006.gguf \
      /mnt/ssd2/V4.1-UD-IQ2_XXS/
   ```

2. Put the folder in the config file's `"env"` (`abyss-<name>.json`; where it is: [SETTINGS](SETTINGS.md)).

   ```json
   "env": {"STRATA_GLM_MIRROR": "/mnt/ssd2/V4.1-UD-IQ2_XXS"}
   ```

   To try it without changing the config: `STRATA_GLM_MIRROR=/mnt/ssd2/V4.1-UD-IQ2_XXS ./abyss.sh --bench --config abyss-<name>.json`.

3. Restart the server. The engine log names the shards it found:

   ```
   [mirror] DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00002-of-00006.gguf also read from /mnt/ssd2/V4.1-UD-IQ2_XXS/...
   ```

   When the engine stops, its log says how much came from each drive:

   ```
   glm fast mirror CUDA0: 115.01 GB from the shards' drive, 68.46 GB from the mirror (37.3%)
   ```

To see whether it helps on your PC, run `./abyss.sh --bench` with and without `STRATA_GLM_MIRROR` ([BENCH](BENCH.md)).

## Space on the second drive

The shards that hold experts are enough. DeepSeek V4.1's Engram table (one shard of its own, 111-135 GB) is read 48 rows a token, so
leaving it out barely changes the speed.

| Model | Space on drive 2 (about) |
| --- | --- |
| DeepSeek V4.1 unsloth UD-IQ2_XXS | 165 GB (shards 1-5) |
| DeepSeek V4.1 unsloth UD-IQ1_M | 140 GB (shards 1-4) |

Both fit on a **256 GB SSD**, so a small NVMe taken out of a laptop works. A smaller one works too with some shards only
(e.g. two of UD-IQ2_XXS's 50 GB shards on 128 GB): only the copied shards' reads are split, so the gain is smaller
(not measured).

## How it works

- At start, for each shard on drive 1, Abyss looks for a file of the same name on drive 2 and uses it only when the size
  and the first and last MiB match (so a different file is never read).
- For every read it picks the drive likely to finish first, from each drive's reads in flight and its recent speed,
  measured again on every read. Drives of different speeds work: the faster one takes more.
- One read in 32 goes to the drive not picked, so a drive that looked slow once is measured again instead of being left
  unused for the rest of the run.
- A failed read from drive 2 is read again from drive 1, and drive 2 is then left out: losing drive 2 does not stop the
  answer.
- It works for every model whose experts stream from the SSD (DeepSeek V4 / V4.1, GLM-5.3-Flash, ...), for the reads
  while answering and for the start-up fill of RAM and VRAM.

## Where it helps and where it does not

| Helps | Helps little |
| --- | --- |
| little RAM (64 GB), much of the model on the SSD | the model nearly fits in memory (a small SSD wait to begin with) |
| both drives in CPU-attached M.2 slots, similar speeds | a slow second drive (SATA, USB 3.x) |
| a slowish first drive (3-4 GB/s) | |
| a big quant (the more it exceeds VRAM + RAM, the better) | a model that fits in VRAM + RAM (Qwen3.8, DeepSeek V4 0731 UD-Q2_K_XL, ...) |

Only the SSD wait gets shorter, so the gain depends on how much of a token is SSD wait. On the same 32 GB + 96 GB, V4.1
UD-IQ1_M (250 GB) should gain a few percent, UD-IQ2_XXS (300 GB) +4.5%, and bigger quants should gain more. A few percent
adds up over long answers of thousands of tokens, and the output does not change.

- On this PC (a Ryzen AI Max+ 395 mini PC) drive 2 is on USB4, sharing the APU's paths with the GPU (OCuLink) and the
  iGPU, and it still helped. With other heavy writes going on (downloads), though, two drives were slower in one test:
  stop other disk traffic when you measure.
- With `STRATA_GLM_DISK_LANE=1` (experts read from the SSD computed by the iGPU) it got slower with one drive and with two
  (this PC: 49 → 50 ms, 51 → 64 ms). Do not combine them.

## Against RAID 0

| | RAID 0 | STRATA_GLM_MIRROR |
| --- | --- | --- |
| Setup | rebuild two drives into one (OS and file system level) | copy files to drive 2 |
| One drive fails | nothing can be read | runs on the other |
| Drives of different speeds | held back by the slower | reads split by speed |
| Space | the sum of both | drive 2 holds a copy |

## Other engines

As far as we checked (Strata v0.2.7.1, Project Maya v1.1.36, llama.cpp, ik_llama.cpp, KTransformers; 2026-10-29), none
reads the same file from two SSDs while answering.

- Project Maya's `MAYA_DS4_LOAD_INTERLEAVE` reads shards **split** over two drives during the start-up copy into RAM;
  it does not touch the SSD wait while answering.
- Strata's `STRATA_MIRROR_MIB` keeps copies of experts in RAM; it is not about SSDs.

## Limits

- Linux only (used by the O_DIRECT reads). On Windows the setting is ignored and drive 1 alone is read.
- Drive 2's files must be the same revision as drive 1's. When the publisher replaces the files, copy them again (a
  different size or first / last MiB logs a warning, and drive 2 is not used).
- Drive 2 needs the space of the shards you copy.
- Several folders can be given, separated by `:`; each shard uses the first copy found (shards can be spread over a
  second and a third drive, but one shard is not read from three or more drives). A third drive of the same speed would cut the SSD wait from a
  half to a third only - less than one to two drives - and a chipset M.2 or the GPU's PCIe link (~7 GB/s on an OCuLink /
  USB4 x4; two drives filled at 7.6 GB/s on this PC) caps it first, so Abyss stops at two.
