#!/usr/bin/env python3
"""Abyss - set up DeepSeek V4 Flash / V4.1 Flash 1017 (run through ./setup-deepseek.sh, which makes the .venv first).

    ./setup-deepseek.sh                      everything below; each step is skipped when it is already done
    ./setup-deepseek.sh --quant UD-IQ3_XXS   another quant of unsloth/DeepSeek-V4-Flash-0731-GGUF
    ./setup-deepseek.sh --quant 32G96G       zephel01's V4 0731 GGUFs: 16G64G, 32G64G or 32G96G (the PC size)
    ./setup-deepseek.sh --model v41          DeepSeek V4.1 Flash 1017: zephel01's LDLQ GGUF for a 32 GB GPU + 96 GB RAM
                                             (--quant 132G, the default: speed; --quant 139G: quality)
    ./setup-deepseek.sh --model ornith       Ornith-1.5-122B, zephel01's IQ3_XXS-B for a 16 GB GPU + 64 GB RAM
                                             (--quant 24G64G: the IQ4_XS file for a 24-32 GB GPU + 64 GB RAM)
    ./setup-deepseek.sh --model ornith35     Ornith-1.5-35B-A3B, the official Q4_K_M for an 8-16 GB GPU + 32 GB RAM
    ./setup-deepseek.sh --model laguna       Laguna-S-2.1 (poolside, 118B-A8B), unsloth's UD-IQ4_XS for a 16-32 GB GPU + 64 GB RAM
    ./setup-deepseek.sh --model inkling2mini  Inkling 2 Mini (Thinking Machines, 128x1.8B looped MoE), unsloth's UD-Q4_K_XL
                                             for a 12-32 GB GPU + 32 GB RAM (text only)
    ./setup-deepseek.sh --model inkling2mini-abl  the same model abliterated, orcarouter's Mix-16G (--quant Mix-12G: the
                                             11.6 GB file) for a 12-24 GB GPU + 32 GB RAM (text only)
    ./setup-deepseek.sh --model ling32air   Ling-3.2-air (inclusionAI, 512 experts, KDA + MLA; BETA, images too), bartowski's
                                             Q4_K_M for a 24-32 GB GPU + 64-96 GB RAM
    ./setup-deepseek.sh --model qwen38-27b   Qwen3.8-27B (dense), unsloth's UD-Q4_K_XL for an 8 GB+ GPU (FFN partly in RAM under ~20 GB)
    ./setup-deepseek.sh --model bonsai27b    Ternary Bonsai 2.5 27B (prism-ml, ternary Qwen3.8-27B, BETA): PTQ1_0 6.2 GB for an 8 GB+ GPU
    ./setup-deepseek.sh --model qwen35-9b-qsa  Qwen3.5-9B QSA (dense, zephel01's Q4_K_M with a QSA indexer) for an 11 GB+ GPU
    ./setup-deepseek.sh --model ornith-9b-qsa  Ornith-1.5-9B QSA (dense, zephel01's Q4_K_M with a QSA indexer) for an 11 GB+ GPU
    ./setup-deepseek.sh --model mimo-9b     MiMo-V2.6-Distill-Qwen-9B (dense, bartowski's Q4_K_M) for an 11 GB+ GPU
    ./setup-deepseek.sh --model mimo-9b-qsa  MiMo-V2.6-Distill-Qwen-9B QSA (zephel01's Q4_K_M with Qwen3.5-9B QSA's indexer) for an 11 GB+ GPU (128K: 12 GB)
    ./setup-deepseek.sh --model sensenova-loop-9b  SenseNova-Loop-9B (looped dense, guideboardlabs' Q4_K_M) for a 10 GB+ GPU
    ./setup-deepseek.sh --model qwen35       Qwen3.5-122B-A10B, zephel01's IQ3_XXS-B2 for a 16 GB GPU + 64 GB RAM
    ./setup-deepseek.sh --model qwen38       Qwen3.8-Flash-Next, zephel01's GGUFs: --quant 12G32G (12 GB GPU + 32 GB RAM),
                                             16G64G (the default) or 24G64G
    ./setup-deepseek.sh --model glm          GLM-5.3-Flash, zephel01's GGUFs: --quant 32G96G (the default) or 32G128G
    ./setup-deepseek.sh --model memt-flash   MemT-1.0-Flash (MemTensor, DeepSeek V4 Flash 0731 with a native memory; BETA,
                                             NVIDIA only): zephel01's 9.3 GB overlay over the v4 --quant 32G96G (L97)
                                             shards, which it sets up too (kept when already there), for a 32 GB GPU +
                                             96 GB RAM
    ./setup-deepseek.sh --model memt-nano    MemT-1.0-Nano (MemTensor, Qwen3.6-35B-A3B with a native memory; BETA, NVIDIA
                                             only): zephel01's Q4_K_M (22.1 GB), measured on a 32 GB GPU (RTX 5090)
    ./setup-deepseek.sh --tools-only         only (re)create the Abyss-tools folder
    (no --quant: the PC-size file that suits this PC's VRAM + RAM is proposed - qwen38, glm, ornith; v4 keeps
     UD-Q2_K_XL and names zephel01's file for this size)
    ./setup-deepseek.sh --dspark             also DeepSeek's DSpark drafter (6.9 GB): faster answers (--no-dspark: off)
    ./setup-deepseek.sh --model qwen38 --dflash   also the DFlash 2 drafter (0.65 GB): faster code (--no-dflash: off)
    ./setup-deepseek.sh --model qwen38-27b   sets up Qwen3.8-27B's DFlash 2 drafter (1.14 GB) too where it fits in VRAM
                                             (a 24 GB+ GPU at UD-Q4_K_XL): faster code / reasoning (--no-dflash: off)

What it does:

  1. checks the PC: an NVIDIA GPU, nvcc, RAM
  2. Python packages into .venv; llama.cpp's source at the pinned commit (third_party/llama.cpp) - for the GPU
     engine's prompt MMQ kernels, which still compile ggml-cuda's sources, and for images (llama.cpp's mtmd)
  3. compiles the engine (build/abyss) for the GPU - arch 120a on RTX 50, whose MXFP4 kernels need it
  4. the model: unsloth's DeepSeek-V4-Flash-0731-GGUF quant into <data>/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<quant>
     (--model v41: zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF, 132G or 139G)
     (--model memt-flash: L97's 3 shards into v4's 32G96G folder, the overlay's shard 1 into its own, and L97's
     shards linked beside it as shards 2-4 - a hard link or a copy where symlinks are not allowed)
     (asks first, or --download-model), every file checked against Hugging Face's SHA-256
  5. the pack next to the shards (tools/iq_pack.py --compat-bf16)
  6. the run config abyss-<name>.json and the start script run-abyss-<name>.sh in the Abyss folder
  7. the DSpark drafter (V4 only; asked when the config is new, or --dspark): bleysg's GGUF of DeepSeek's block drafter into
     <data>/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF, its path in the config's env (STRATA_DS4_DSPARK)
     (inkling2mini / inkling2mini-abl: unsloth's dspark-Inkling-2-Mini-Q8_0.gguf, not downloaded - when it is on
     disk the engine finds and keeps it if the experts fit beside it; --no-dspark: STRATA_INK_DSPARK=0 in the env)
  8. the Abyss-tools folder (abyss.sh start / stop / status / update / log / models) from ops/abyss-tools/

The data folder defaults to Abyss-data beside Abyss, the tools folder to Abyss-tools beside it.  The helpers (the PC
checks, pip, llama.cpp's source, resumable downloads) are setup.py's, inherited from Maya and Strata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIN = os.name == "nt"
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
sys.path.insert(0, str(ROOT / "tools"))
import prebuilt as P  # noqa: E402  the prebuilt CUDA engine (no toolkit / compiler needed)

REPO = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
# Qwen3.8-27B UD-Q4_K_XL, the whole file in VRAM: the GPU size (GB, as the card reports it) each context needs.  RTX 5090
# with VRAM held back (2026-11-12): the engine takes 19.0 GB at 8K, 20.7 at 32K, 22.8 at 64K, ~27.7 at 128K (F16 KV,
# NextN drafter, a 4K prompt); short of that it stops at the start, or at the first prompt (20 GB free, 32K).  ~1 GB
# left for the desktop and the driver: a 24 GB card gets 32K (64K ran with nothing else on it), a 32 GB one 128K
DENSE27_VRAM = {8192: 21.0, 32768: 23.5, 65536: 26.0, 131072: 30.0, 262144: 40.0}   # (256K: +8.7 GB of caches, estimated)
# a smaller GPU (2026-11-12, v1.3.139): the engine keeps the FFN rows that do not fit in pinned RAM and computes them on
# the CPU; the GPU then needs the rest of the file, the caches and two staging buffers - the table less the FFN (9.9 GB)
# plus 0.4 GB (RTX 5090 capped: 8K started at 9.0 GB with no FFN in VRAM; 12 GB 27 tok/s, 16 GB 65, 20 GB all in VRAM)
DENSE27_FFN_GB = 9.5
# an 8 GB card (v1.3.142): the engine also keeps the head's last rows on the CPU and loads its kernels lazily - 8K
# only (RTX 5090 capped to 8 GB, 8K, the drafter: peak 7.8 GB, 15.7 tok/s; llama.cpp 8.0) with nothing else on the GPU
DENSE27_SMALL = (7.5, 8192)   # (the GPU size it takes as reported, the context it gets)
DENSE27_RAM_GB = 32   # (up to 10 GB of FFN pinned + the 6 GB headroom + the server)
# every file of unsloth/Qwen3.8-27B-GGUF the engine runs (2026-11-14): its size in GB and whether it carries the NextN
# draft block - the UD-IQ1 / UD-IQ2 files do not, and the setup fetches unsloth's separate one (MTP/, 1.4 GB) for them.
# The VRAM a context needs moves with the file's size (the table above is UD-Q4_K_XL's), the FFN part with its share
DENSE27_QUANTS = {
    "UD-IQ1_S": (6.19, False), "UD-IQ1_M": (6.73, False), "UD-IQ2_XXS": (7.21, False), "UD-IQ2_XS": (7.63, False),
    "UD-IQ2_S": (8.42, False), "UD-Q2_K_XL": (9.95, True), "UD-IQ3_XXS": (10.93, True), "UD-IQ3_S": (12.04, True),
    "UD-Q3_K_XL": (13.15, True), "UD-IQ4_XS": (14.35, True), "UD-Q4_K_S": (15.36, True), "Q4_0": (16.06, True),
    "UD-Q4_K_M": (16.56, True), "Q4_1": (17.54, True), "UD-Q4_K_XL": (17.66, True), "UD-Q5_K_S": (18.76, True),
    "UD-Q5_K_M": (19.87, True), "UD-Q5_K_XL": (20.98, True), "UD-Q6_K": (21.98, True), "UD-Q6_K_M": (23.09, True),
    "UD-Q6_K_L": (24.18, True), "UD-Q6_K_XL": (25.30, True), "UD-Q8_K_L": (28.05, True), "Q8_0": (29.05, True),
    "UD-Q8_K_XL": (31.46, True), "BF16": (54.66, True)}
# the files the PC-size proposal picks among (the largest that holds an 8K context all in VRAM)
DENSE27_PICKS = ("UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q4_K_XL", "UD-Q6_K", "Q8_0")
# unsloth's separate NextN block for the files without one, and the image encoder's weights (Qwen3.8-27B reads images)
DENSE27_MTP = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
DENSE27_MMPROJ = "mmproj-F16.gguf"
LING32_MMPROJ = "mmproj-Ling-3.2-air-f16.gguf"   # (bartowski's repo, top level: 0.87 GB, the vision tower + projector)
# Ternary Bonsai 2.5 27B (prism-ml, Apache-2.0; BETA): Qwen3.8-27B with ternary weights (one FP16 scale a 128) in a
# blockwise Hadamard basis (prism.hadamard.* in the GGUF: the engine rotates the activations, prism_path.cu).  Two files
# at the repo's top level: PTQ1_0 (6.23 GB, with the MTP block blk.64) and Q2_0 (7.67 GB, the same weights, no MTP).
# The GPU size each context needs, PTQ1_0, MTP on, 4K prompt (bench/bonsai/vram.txt, 2026-11-25, RTX 5090 capped:
# a cap short of it puts rows of 52 layers on the CPU, 4K prompt then 30.5 tok/s at 32K / 9 GB, 64.8 at 64K / 11 GB):
# at 8 / 9.5 / 11.75 / 16 GB for 8K / 32K / 64K / 128K only one layer's last 1,024 rows are on the CPU - decode 178.6
# (8K, median of 3) / 179.4 / 186.9 / 186.4 tok/s (one run each; bench/bonsai/vram2.txt), against 185.3 uncapped
BONSAI27_VRAM = {8192: 8.0, 32768: 9.5, 65536: 12.0, 131072: 16.0}
BONSAI27_QUANTS = {"PTQ1_0": (6.23, True), "Q2_0": (7.67, True)}   # (Q2_0 has no draft block: the engine runs without)
BONSAI27_FFN_GB = 3.7   # (64 layers x 3 x 5120 x 17408 trits at 28 bytes a 128)
# Qwen3.5-9B QSA (zephel01's, BETA, v1.3.162): the GPU size each context needs.  Measured on an RTX 5090, Q4_K_M, MTP
# drafts on (bench/train9b/results.md, 2026-11-17): with no cap the engine peaks at 11.17 / 12.40 / 14.86 GB at 32K / 64K /
# 128K, 4 GB of it the prompt path's buffer, which it sizes from the free VRAM; capped with STRATA_GLM_VRAM_GB=15 a 128K
# context peaks at 11.65 GB, capped at 11 GB at 10.68 GB (0.77 GB of the FFN in RAM), at the same speed (decode 245
# tok/s).  So the table is that 11 GB cap, for every context up to 128K; longer ones are not measured.  Q8_0 is not
# measured for memory: it is taken to need its file size more (+4.0 GB, as dense_shift does for any other file)
QSA9_VRAM = {131072: 11.0}
QSA9_QUANTS = {"Q4_K_M": (5.85, True), "Q8_0": (9.86, True)}   # (GB; each file carries its NextN block, blk.32)
# Ornith-1.5-9B QSA ("ornith-9b-qsa") has the same shapes, indexer layout and uncapped peaks (GPU 11.17 / 12.40 / 14.86 GB at
# 32K / 64K / 128K) as Qwen3.5-9B QSA, so it uses QSA9_VRAM / gpu_min 11 from Qwen3.5-9B QSA's capped run (a VRAM cap was not
# measured on Ornith itself).  Files: Q4_K_M 5,853,504,192 bytes, Q8_0 9,859,474,112 bytes
# MiMo-V2.6-Distill-Qwen-9B ("mimo-9b", v1.3.168): Xiaomi's SFT of Qwen3.5-9B (MIT), bartowski's GGUFs.  Same shapes as
# Qwen3.5-9B, no NextN block in the files.  Measured on an RTX 5090, Q4_K_M, no cap (bench/mimo9b/results.md,
# 2026-11-17): GPU peaks 11.17 / 14.27 GB at 32K / 128K (15.11 GB at 128K with the borrowed draft block) - the same
# as Qwen3.5-9B's (11.33 GB at 32K), so the table is QSA9_VRAM (Qwen3.5-9B's 11 GB cap at 128K; a cap was not measured
# on MiMo itself).  Q8_0 is not measured: dense_shift adds its size.  Files: Q4_K_M 5,841,049,120 bytes, Q8_0
# 9,545,979,424 bytes
MIMO9_QUANTS = {"Q4_K_M": (5.84, True), "Q8_0": (9.55, True)}
# the draft block: Qwen3.5-9B's own NextN block (blk.32 of Qwen's Q4_K_M, 0.16 GB, Apache-2.0) drafts for MiMo with 76-78%
# accepted (decode 217 -> 273 tok/s at 4K, 186 -> 228 at 32K; without it MiMo drafts with n-grams only).  Fetched into
# MTP/ beside the GGUF, where the engine takes it by itself; when the repo does not have it (yet), MiMo runs without it
MIMO9_MTP_REPO = "zephel01/MiMo-V2.6-Distill-Qwen-9B-MTP-GGUF"
MIMO9_MTP = "mtp-Qwen3.5-9B-Q4_K_M.gguf"
# MiMo-V2.6-Distill-Qwen-9B QSA ("mimo-9b-qsa", BETA): bartowski's MiMo GGUFs with Qwen3.5-9B QSA's indexer added as it is
# (the same 7 layers, layer 3 dense, top_k 4096; not trained on MiMo).  RTX 5090, borrowed draft block on (bench/mimo9b/
# results.md, 2026-11-18): GPU peaks 11.67 / 15.36 GB at 32K / 128K uncapped (Q4_K_M).  Capped with STRATA_GLM_VRAM_GB, Q4_K_M:
# 32K at 11 GB all in VRAM (268 tok/s); 128K at 11 GB puts 1.32 GB of FFN in RAM (145 tok/s against 208 uncapped), at 12 GB
# one layer's last 256 rows (206 tok/s), at 13 GB none; Q8_0 128K at 15 GB 181 tok/s (= uncapped), which the +3.71 GB
# shift (dense_shift) covers.  Unlike Qwen3.5-9B QSA (128K at the same speed at 11 GB) the borrowed draft block and its
# verify window take the room, so the table is 11 GB for 32K, 12 GB for 128K.
# Files: Q4_K_M 5,914,462,912 bytes, Q8_0 9,619,393,216 bytes
MIMO9Q_VRAM = {32768: 11.0, 131072: 12.0}
MIMO9Q_QUANTS = {"Q4_K_M": (5.91, True), "Q8_0": (9.62, True)}
# SenseNova-Loop-9B ("sensenova-loop-9b", arch sensenova_loop): a looped Qwen3.5-9B (32 blocks run as 56 layer visits - blocks 4-27
# once more after the first pass, each visit with its own KV rows and linear-attention state - with two-stream mHC around
# the loop), guideboardlabs' GGUFs (Apache-2.0).  No NextN block: n-gram drafts.  Measured on an RTX 5090, Q4_K_M
# (bench/loop9b/results.md, 2026-11-18): with no cap the engine peaks at 10.2 / 12.0 / 13.8 / 17.3 GB at 8K / 32K / 64K /
# 128K (the prompt path's buffer sized from the free VRAM); capped with STRATA_GLM_VRAM_GB at 10 / 12 / 14 GB a 32K / 64K
# / 128K context peaks at 8.4 / 10.2 / 13.8 GB at the same decode speed (96.8 / 79.0 / 62.4 tok/s).  So the table is those
# caps; the other files are taken to need their size more or less (dense_shift)
LOOP9_VRAM = {32768: 10.0, 65536: 12.0, 131072: 14.0}
LOOP9_QUANTS = {"Q3_K_M": (4.64, True), "Q4_0": (5.33, True), "Q4_K_M": (5.63, True), "Q8_0": (9.54, True)}   # (GB)
# the models this setup installs: the repo, its default quant (a folder of the repo), the quants it offers when the repo
# is a fixed set, the config's model_name, the config name's prefix, the env the model's measurements were made with,
# the experts' size in GB at the default quant (for the RAM line)
# bartowski/Ling-3.2-air-GGUF's quants that are a folder of shards (the rest: one file at the repo's top level)
LING32_FOLDERS = ("IQ3_XXS", "IQ3_XS", "IQ3_M", "IQ4_XS", "IQ4_NL", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1",
                  "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q5_K_S", "Q5_K_M", "Q6_K_S", "Q6_K", "Q8_0", "bf16")

MODELS = {
    "v4": {"repo": REPO, "quant": "UD-Q2_K_XL", "quants": None, "model_name": "deepseek-v4-flash", "prefix": "dsv4",
           "title": "DeepSeek V4 Flash 0731", "env": {}, "gb": 97,
           # --quant 16G64G / 32G64G / 32G96G: zephel01's LDLQ GGUFs of the same model (2026-11-08, one repo a PC size;
           # any other --quant is a folder of unsloth's repo)
           "variants": {
               "16G64G": {"repo": "zephel01/DeepSeek-V4-Flash-0731-IQ1_S-16G64G-GGUF", "folder": "L62",
                          "prefix_file": "DeepSeek-V4-Flash-0731-L62-", "gb": 62, "vram": 16, "ram": 64},
               "32G64G": {"repo": "zephel01/DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G-GGUF", "folder": "L72",
                          "prefix_file": "DeepSeek-V4-Flash-0731-L72-", "gb": 72, "vram": 32, "ram": 64},
               "32G96G": {"repo": "zephel01/DeepSeek-V4-Flash-0731-IQ2_M-32G96G-GGUF", "folder": "L97",
                          "prefix_file": "DeepSeek-V4-Flash-0731-L97-", "gb": 97, "vram": 32, "ram": 96}}},
    # zephel01's own LDLQ GGUF (2026-11-02, BETA): 132G = UD-IQ1_M's types with better rounding (speed), 139G = layers
    # 35-39 one step higher (quality).  The env is the model card's everyday settings (its speed and quality rows)
    "v41": {"repo": "zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF", "quant": "132G", "quants": ("132G", "139G"),
            "model_name": "deepseek-v4.1-flash-1017", "prefix": "dsv41", "title": "DeepSeek V4.1 Flash 1017",
            "env": {"STRATA_GLM_DISK_READERS": "8", "STRATA_GLM_TAIL_SKIP": "4", "STRATA_GLM_TAIL_SKIP_DISK": "1",
                    "STRATA_GLM_ROUTE_AFFINITY_RAM": "0.08"}, "gb": 140},
    # zephel01's 16 GB GPU + 64 GB RAM quants of the two Qwen3.5-MoE 122B models (the files at the repo's top level;
    # the quant names the local folder).  Ornith's NextN drafter is in its second shard: the engine reads it there
    "ornith": {"repo": "zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF", "quant": "IQ3_XXS-B", "quants": ("IQ3_XXS-B", "24G64G", "32G128G", "16G64G"),
               "folder": "", "prefix_file": "Ornith-1.5-122B-IQ3_XXS-B-", "model_name": "ornith-1.5-122b",
               "prefix": "ornith", "title": "Ornith-1.5-122B", "env": {}, "gb": 52, "pack_args": [],
               "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
               # (--quant 24G64G: the IQ4_XS file for a 24-32 GB GPU + 64 GB RAM, 2026-11-08)
               "variants": {"IQ3_XXS-B": {"vram": 16, "ram": 64},
                            "24G64G": {"repo": "zephel01/Ornith-1.5-122B-IQ4_XS-24G64G-GGUF",
                                       "prefix_file": "Ornith-1.5-122B-24G64G-", "gb": 69, "vram": 24, "ram": 64},
                            # (--quant 32G128G: Ornith-1.5-397B, 10 experts a token, for a 32 GB GPU + 128 GB RAM, 2026-11-09)
                            "32G128G": {"repo": "zephel01/Ornith-1.5-397B-IQ2_XXS-32G128G-GGUF",
                                        "prefix_file": "Ornith-1.5-397B-32G128G-", "model_name": "ornith-1.5-397b",
                                        "title": "Ornith-1.5-397B", "gb": 114, "vram": 32, "ram": 128},
                            # (--quant 16G64G: Ornith-1.5-397B with a lighter dense part (Q5_K / IQ4_XS) so 128K starts on a
                            # 16 GB GPU + 64 GB RAM, 2026-11-12, #44; ~9 tok/s, SSD-bound - the 122B is ~5x faster)
                            "16G64G": {"repo": "zephel01/Ornith-1.5-397B-IQ2_XXS-16G64G-GGUF",
                                       "prefix_file": "Ornith-1.5-397B-16G64G-", "model_name": "ornith-1.5-397b",
                                       "title": "Ornith-1.5-397B", "gb": 111, "vram": 16, "ram": 64}}},
    # Ornith-1.5-35B-A3B (same qwen35moe engine path): the official GGUFs, one file a quant at the repo's top level.
    # For an 8-16 GB GPU (2026-11-12, RTX 5090 capped: Q4_K_M decode 156 / 209 / 262 tok/s at 8 / 12 / 16 GB, llama.cpp
    # 106 / 133 / 173); NextN drafter from the file
    "ornith35": {"repo": "ornith-ai/Ornith-1.5-35B-A3B-GGUF", "quant": "Q4_K_M", "quants": ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"),
                 "folder": "", "prefix_tmpl": "Ornith-1.5-35B-{q}.", "model_name": "ornith-1.5-35b-a3b",
                 "prefix": "ornith35", "title": "Ornith-1.5-35B-A3B", "env": {}, "gb": 22, "pack_args": [],
                 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},
    # Laguna-S-2.1 (poolside, gguf arch laguna: the qwen35moe engine path's laguna layers): unsloth's GGUFs, a folder a
    # quant (3 shards).  For a 16-32 GB GPU + 64 GB RAM (the experts are 53 GB at UD-IQ4_XS); no drafter in the file
    "laguna": {"repo": "unsloth/Laguna-S-2.1-GGUF", "quant": "UD-IQ4_XS", "quants": ("UD-IQ4_XS",),
               "prefix_tmpl": "Laguna-S-2.1-{q}-", "model_name": "laguna-s-2.1", "prefix": "laguna",
               "title": "Laguna-S-2.1", "env": {}, "gb": 58, "pack_args": [],
               "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 20}},
    # Inkling 2 Mini (Thinking Machines, gguf arch inkling2 - unsloth's own looped layout - or inkling2_loop - its
    # Mainline_Support copy / ggml-org's; the engine reads both): unsloth's GGUFs, one file a quant at the repo's top
    # level.  17.6 GB at UD-Q4_K_XL (NVFP4 experts): all in VRAM on a 24-32 GB GPU, a 12-16 GB GPU + 32 GB RAM holds
    # the rest in the RAM tier.  Text only; no drafter in the file; the prompt runs token by token for now
    "inkling2mini": {"repo": "unsloth/Inkling-2-Mini-GGUF", "quant": "UD-Q4_K_XL", "quants": ("UD-Q4_K_XL",),
                     "folder": "", "prefix_tmpl": "Inkling-2-Mini-{q}.", "model_name": "inkling-2-mini",
                     "prefix": "inkling2mini", "title": "Inkling 2 Mini", "env": {}, "gb": 18, "pack_args": [],
                     "sampling": {"temperature": 1.0, "top_k": 50}},
    # Inkling 2 Mini abliterated (orcarouter, 2026-11-24): the same inkling2_loop layout with mixed stock llama.cpp types
    # (Mix-16G: NVFP4 / IQ4_XS / Q5_K gate_up over IQ4_NL / Q5_1 down; Mix-12G: IQ2_XXS..IQ3_S gate_up over IQ4_NL down),
    # one file a quant at the repo's top level.  All in VRAM on a 16-24 GB GPU; unsloth's drafter fits it too
    "inkling2mini-abl": {"repo": "orcarouter/Inkling-2-Mini-abliterated-GGUF", "quant": "Mix-16G",
                         "quants": ("Mix-16G", "Mix-12G"), "folder": "", "prefix_tmpl": "Inkling-2-Mini-abliterated-{q}.",
                         "model_name": "inkling-2-mini-abliterated", "prefix": "inkling2mini-abl",
                         "title": "Inkling 2 Mini abliterated", "env": {}, "gb": 16, "pack_args": [],
                         "sampling": {"temperature": 1.0, "top_k": 50}},
    # Ling-3.2-air (inclusionAI, gguf arch bailingmoe3: the qwen35moe engine path's ling layers - KDA + MLA, 512 experts,
    # BETA; images with bartowski's mmproj and abyss-vision): bartowski's GGUFs, a folder of shards a quant (the small
    # ones one file at the repo's top level).
    # 80.4 GB at Q4_K_M: a 24-32 GB GPU + 64-96 GB RAM holds the experts in VRAM + RAM.  No MTP drafter yet; the file's
    # rope.dimension_sections (llama.cpp's VL conversion) is ignored - the engine rotates the model's interleaved pairs.
    # 2026-11-24, RTX 5090 + RAM: decode 171 tok/s after a 4K prompt, prompts 2.2K tok/s at 4K, 3.3K at 16K
    "ling32air": {"repo": "bartowski/Ling-3.2-air-GGUF", "quant": "Q4_K_M", "quants": ("Q4_K_M",),
                  # (--quant <any of the repo's files> resolves; Q4_K_M is the one checked here: its folder of shards,
                  # as the other folder quants - IQ1_S .. Q2_K are single files at the top level)
                  "quant_folders": {q: "Ling-3.2-air-" + q for q in LING32_FOLDERS}, "folder": "",
                  "prefix_tmpl": "Ling-3.2-air-{q}.",
                  "model_name": "ling-3.2-air", "prefix": "ling32air", "title": "Ling-3.2-air", "env": {}, "gb": 81,
                  "pack_args": [], "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}},
    # Qwen3.8-27B (a dense qwen35: every layer a dense FFN, no experts): unsloth's GGUFs, one file a quant at the repo's
    # top level, NextN drafter from the file.  This first version runs the whole file in one GPU's VRAM (no RAM tier,
    # no layer split): "vram" is the GPU size each context needs (dense_context).  2026-11-12, RTX 5090, UD-Q4_K_XL:
    # decode 131.6 tok/s with the drafter, 72.7 without (llama.cpp 72.1)
    "qwen38-27b": {"repo": "unsloth/Qwen3.8-27B-GGUF", "quant": "UD-Q4_K_XL", "quants": tuple(DENSE27_QUANTS),
                   "quant_gb": DENSE27_QUANTS, "picks": DENSE27_PICKS, "quant_folders": {"BF16": "BF16"},
                   "folder": "", "prefix_tmpl": "Qwen3.8-27B-{q}.", "model_name": "qwen3.8-27b",
                   "prefix": "qwen38-27b", "title": "Qwen3.8-27B", "env": {}, "gb": 18, "pack_args": [],
                   "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                   "vram_ctx": DENSE27_VRAM, "ffn_gb": DENSE27_FFN_GB, "ram_min": DENSE27_RAM_GB,
                   "small_gpu": DENSE27_SMALL, "gpu_min": 8},
    # Ternary Bonsai 2.5 27B (prism-ml; BETA): a dense qwen35 like qwen38-27b (BONSAI27_VRAM above); text only (no
    # image projector in the repo); the card's sampling.  An 8 GB GPU keeps a part of the FFN on the CPU lane
    "bonsai27b": {"repo": "prism-ml/Ternary-Bonsai-2.5-27B-GGUF", "quant": "PTQ1_0", "quants": tuple(BONSAI27_QUANTS),
                  "quant_gb": BONSAI27_QUANTS, "picks": ("PTQ1_0",), "folder": "",
                  "prefix_tmpl": "Ternary-Bonsai-2.5-27B-{q}", "model_name": "ternary-bonsai-2.5-27b",
                  "prefix": "bonsai27b", "title": "Ternary Bonsai 2.5 27B", "env": {}, "gb": 7, "pack_args": [],
                  "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                  "vram_ctx": BONSAI27_VRAM, "ffn_gb": BONSAI27_FFN_GB, "ram_min": 16, "gpu_min": 8},
    # Qwen3.5-9B QSA (a dense qwen35 like the 27B, with a QSA indexer trained at home; BETA): zephel01's GGUFs, one file
    # a quant at the repo's top level, each with its NextN drafter.  The engine reads the QSA layers from the file (7
    # of 8 attention layers, layer 3 dense: compress_ratios 0) from v1.3.162; no env.  Text only: the file has no
    # image projector.  Whole file in one GPU (QSA9_VRAM above); 2026-11-17, RTX 5090, 128K prompt: decode 240 tok/s
    # (132 dense)
    "qwen35-9b-qsa": {"repo": "zephel01/Qwen3.5-9B-QSA-GGUF", "quant": "Q4_K_M", "quants": tuple(QSA9_QUANTS),
                      "quant_gb": QSA9_QUANTS, "picks": tuple(QSA9_QUANTS), "folder": "",
                      "prefix_tmpl": "Qwen3.5-9B-QSA-{q}.", "model_name": "qwen3.5-9b-qsa", "prefix": "qwen35-9b-qsa",
                      "title": "Qwen3.5-9B QSA", "env": {}, "gb": 6, "pack_args": [],
                      "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                      "vram_ctx": QSA9_VRAM, "gpu_min": 11},
    # Ornith-1.5-9B QSA (v1.3.166, BETA): the same as qwen35-9b-qsa (Qwen3.5-9B QSA) with Ornith's weights and its own chat
    # template in the GGUF.  The VRAM table / gpu_min are Qwen3.5-9B QSA's (same shapes, same uncapped peaks).  Sampling: the
    # base model ornith-ai/Ornith-1.5-9B has no generation_config.json, so the qwen35 values (= the model card's coding
    # preset, the same as the other Ornith entries).  Text only
    "ornith-9b-qsa": {"repo": "zephel01/Ornith-1.5-9B-QSA-GGUF", "quant": "Q4_K_M", "quants": tuple(QSA9_QUANTS),
                      "quant_gb": QSA9_QUANTS, "picks": tuple(QSA9_QUANTS), "folder": "",
                      "prefix_tmpl": "Ornith-1.5-9B-QSA-{q}.", "model_name": "ornith-1.5-9b-qsa", "prefix": "ornith-9b-qsa",
                      "title": "Ornith-1.5-9B QSA", "env": {}, "gb": 6, "pack_args": [],
                      "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                      "vram_ctx": QSA9_VRAM, "gpu_min": 11},
    # MiMo-V2.6-Distill-Qwen-9B (v1.3.168): bartowski's GGUFs, one file a quant at the repo's top level; the chat template
    # (MiMo v2.6's: the model opens <think> itself) and the sampling (temperature 0.6, top_p 0.95, top_k 20) are the
    # GGUF's.  Drafts with Qwen3.5-9B's NextN block (MIMO9_MTP) when it can be fetched.  Text only
    "mimo-9b": {"repo": "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF", "quant": "Q4_K_M", "quants": tuple(MIMO9_QUANTS),
                "quant_gb": MIMO9_QUANTS, "picks": tuple(MIMO9_QUANTS), "folder": "",
                "prefix_tmpl": "MiMo-V2.6-Distill-Qwen-9B-{q}.", "model_name": "mimo-v2.6-distill-qwen-9b",
                "prefix": "mimo-9b", "title": "MiMo-V2.6-Distill-Qwen-9B", "env": {}, "gb": 6, "pack_args": [],
                "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                "vram_ctx": QSA9_VRAM, "gpu_min": 11},
    # MiMo-V2.6-Distill-Qwen-9B QSA (BETA): zephel01's GGUFs, one file a quant at the repo's top level - mimo-9b's files
    # with Qwen3.5-9B QSA's indexer grafted on.  The engine reads the QSA layers from the file (as qwen35-9b-qsa); drafts
    # with the same borrowed block as mimo-9b; MiMo's chat template and sampling.  Text only
    "mimo-9b-qsa": {"repo": "zephel01/MiMo-V2.6-Distill-Qwen-9B-QSA-GGUF", "quant": "Q4_K_M", "quants": tuple(MIMO9Q_QUANTS),
                    "quant_gb": MIMO9Q_QUANTS, "picks": tuple(MIMO9Q_QUANTS), "folder": "",
                    "prefix_tmpl": "MiMo-V2.6-Distill-Qwen-9B-QSA-{q}.", "model_name": "mimo-v2.6-distill-qwen-9b-qsa",
                    "prefix": "mimo-9b-qsa", "title": "MiMo-V2.6-Distill-Qwen-9B QSA", "env": {}, "gb": 6,
                    "pack_args": [], "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                    "vram_ctx": MIMO9Q_VRAM, "gpu_min": 11},
    # SenseNova-Loop-9B: guideboardlabs' GGUFs, one file a quant at the repo's top level; the chat template and the sampling
    # (temperature 1.0, top_p 0.95: general.sampling in the GGUF) are the file's.  One GPU only (the engine runs the loop
    # on one device).  Text only
    "sensenova-loop-9b": {"repo": "guideboardlabs/SenseNova-Loop-9B-GGUF", "quant": "Q4_K_M", "quants": tuple(LOOP9_QUANTS),
                "quant_gb": LOOP9_QUANTS, "picks": tuple(LOOP9_QUANTS), "folder": "",
                "prefix_tmpl": "SenseNova-Loop-9B-{q}.", "model_name": "sensenova-loop-9b",
                "prefix": "sensenova-loop-9b", "title": "SenseNova-Loop-9B", "env": {}, "gb": 6, "pack_args": [],
                "sampling": {"temperature": 1.0, "top_p": 0.95},
                "vram_ctx": LOOP9_VRAM, "gpu_min": 10},
    # MemTensor's MemT-1.0-Nano (Apache-2.0; native memory, BETA, v1.3.204): Qwen3.6-35B-A3B (the qwen35moe path, Ornith-
    # 1.5-35B-A3B's shapes) with a memory block in each of its 10 full-attention layers.  zephel01's Q4_K_M (tools/
    # memt_convert.py, Ornith's per-tensor types; the memory blocks BF16 / F32), one file at the repo's top level.
    # NVIDIA only, one GPU; no drafter (MemT has no MTP weights).  Measured on an RTX 5090 (32 GB) only, all in VRAM:
    # decode 304.8 tok/s without a memory, 279.8 with one (4K prompt + 512, median of 3); a smaller GPU is not measured.
    # 16K context and the generation_config's sampling, as the manual's config
    "memt-nano": {"repo": "zephel01/MemT-1.0-Nano-Q4_K_M-GGUF", "quant": "Q4_K_M", "quants": ("Q4_K_M",), "folder": "",
                  "prefix_tmpl": "MemT-1.0-Nano-{q}.", "model_name": "memt-1.0-nano", "prefix": "memt-nano",
                  "title": "MemT-1.0-Nano", "env": {}, "gb": 22, "pack_args": [],
                  "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
                  "nvidia_only": True, "context": 16384},
    # MemTensor's MemT-1.0-Flash (MIT; native memory, BETA, v1.3.205): DeepSeek V4 Flash 0731 with a memory block in each
    # of its 20 CSA layers.  zephel01's overlay is ONE file, shard 1 of 4 (L97's keys, MemT's own non-expert backbone in
    # L97's types, the memory weights Q8_0; 9.27 GB): shards 2-4 are the v4 --quant 32G96G (L97) shards as they are -
    # MemT's experts are 0731's - so "base" fetches those into v4's own folder (L97 already set up is not downloaded
    # again) and setup links them beside the overlay under its name (iq_pack.py and the engine find the siblings by
    # the -0000N-of-00004 pattern; the pack is --first-wins: shard 1's backbone shadows L97's).  NVIDIA only, one GPU; no
    # drafters (the verify windows do not read the memory).  RTX 5090, 4K prompt + 512: decode 77.5 tok/s memory off,
    # 68.2 on (L97 alone 78.9).  16K context as the manual's config; sampling the default (MemT-1.0-Flash's
    # generation_config was not fetched: the memory blocks and config.json only)
    "memt-flash": {"repo": "zephel01/MemT-1.0-Flash-32G96G-GGUF", "quant": "32G96G", "quants": ("32G96G",), "folder": "",
                   "prefix_file": "MemT-1.0-Flash-32G96G-", "model_name": "memt-1.0-flash", "prefix": "memt-flash",
                   "title": "MemT-1.0-Flash", "env": {}, "gb": 106, "pack_args": ["--compat-bf16", "--first-wins"],
                   "base": ("v4", "32G96G"), "nvidia_only": True, "context": 16384,
                   # MemTensor/MemT-1.0-Flash generation_config.json (2026-11-25): temperature 1.0, top_p 1.0
                   "sampling": {"temperature": 1.0, "top_p": 1.0}},
    "qwen35": {"repo": "zephel01/Qwen3.5-122B-A10B-IQ3_XXS-16G64G-GGUF", "quant": "IQ3_XXS-B2", "quants": ("IQ3_XXS-B2",),
               "folder": "", "prefix_file": "Qwen3.5-122B-A10B-IQ3_XXS-B2-", "model_name": "qwen3.5-122b-a10b",
               "prefix": "qwen35", "title": "Qwen3.5-122B-A10B", "env": {}, "gb": 51, "pack_args": [],
               "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}},
    # zephel01's Qwen3.8-Flash-Next and GLM-5.3-Flash GGUFs: one repo a PC size, so each quant (the size) names its
    # repo, the folder inside it and the shards' file prefix ("variants").  Qwen3.8's MTP drafter is in the shards
    "qwen38": {"quant": "16G64G", "quants": ("12G32G", "16G64G", "24G64G"), "model_name": "qwen3.8-flash-next",
               "prefix": "qwen38", "title": "Qwen3.8-Flash-Next", "env": {}, "pack_args": ["--fast"],
               "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
               "variants": {
                   "12G32G": {"repo": "zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF", "folder": "L448",
                              "prefix_file": "Qwen3.8-Flash-Next-12G32G-L448-", "gb": 67, "vram": 12, "ram": 32},
                   "16G64G": {"repo": "zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF", "folder": "",
                              "prefix_file": "Qwen3.8-Flash-Next-16G64G-", "gb": 87, "vram": 16, "ram": 64},
                   "24G64G": {"repo": "zephel01/Qwen3.8-Flash-Next-IQ3_M-24G64G-GGUF", "folder": "",
                              "prefix_file": "Qwen3.8-Flash-Next-24G64G-", "gb": 94, "vram": 24, "ram": 64}}},
    "glm": {"quant": "32G96G", "quants": ("32G96G", "32G128G"), "model_name": "glm-5.3-flash", "prefix": "glm",
            "title": "GLM-5.3-Flash", "env": {}, "sampling": {"temperature": 1.0, "top_p": 0.95},
            "variants": {
                "32G96G": {"repo": "zephel01/GLM-5.3-Flash-IQ2_XS-32G96G-GGUF", "folder": "S1",
                           "prefix_file": "GLM-5.3-Flash-S1-", "gb": 94, "vram": 32, "ram": 96},
                "32G128G": {"repo": "zephel01/GLM-5.3-Flash-IQ3_XXS-32G128G-GGUF", "folder": "Q120d",
                            "prefix_file": "GLM-5.3-Flash-Q120d-", "gb": 128, "vram": 32, "ram": 128}}},
}
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{path}"
HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main/{folder}"
# the quant this setup was tested with; its files are known even without the Hugging Face API
KNOWN = {
    "UD-Q2_K_XL": [
        ("UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf", 5257664,
         "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b"),
        ("UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf", 49437013568,
         "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327"),
        ("UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf", 47390237120,
         "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4"),
    ],
}
# DeepSeek's DSpark block drafter for V4-Flash-0731 (v1.3.43): bleysg's GGUF of the official drafter (3 layers over the
# target's layers 40-42, Q2_K experts, the Markov head in Q8_0; MIT), pinned to the commit it was measured with.  The
# engine loads it from the config's STRATA_DS4_DSPARK (src/core/dspark.cu) and takes ~6.5 GB of VRAM for it
DSPARK_REPO = "bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF"
DSPARK_REV = "57fae967d0c5dac1f32a444ea8884ec46459247b"
DSPARK_FILE = ("DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf", 6940214304,
               "d2025be3a4351f20d9f3d4c58eda9d8577cfda878bee7377bbb71af2087d9181")
DSPARK_URL = "https://huggingface.co/{repo}/resolve/{rev}/{file}"
# Qwen3.8-Flash-Next's DFlash 2 block drafter (v1.3.125, opt-in): nazeshinjite's Q8_0 GGUF of incoai's drafter (5 dense
# layers over the target's layers 6/16/26/36/46; Apache-2.0), pinned to the commit it was measured with.  The engine
# loads it from the config's STRATA_Q4X_DFLASH (src/core/dflash2.cu, ~0.76 GB of VRAM) in place of the MTP layer
DFLASH_REPO = "nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF"
DFLASH_REV = "cbe1b23f1466458b9d839f6cbaef7d3b52d8cc9b"
DFLASH_FILE = ("Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf", 652101056,
               "5e7163abcbee4f348adced14c0f545fa4f2c5a1fdfd4d0a914b176e1c866ac9e")
# Qwen3.8-27B's DFlash 2 block drafter (v1.3.171, on by default where it fits): z-lab's Q4_K_M GGUF (5 layers over the
# target's layers 6/20/34/48/62; Apache-2.0), pinned to the commit it was measured with.  The engine loads it from the
# config's STRATA_Q35_DFLASH (src/core/dflash2.cu, ~1.4 GB of VRAM) beside the NextN block and drops it when the FFN
# would not all fit in VRAM beside it
DFLASH27_REPO = "z-lab/Qwen3.8-27B-DFlash2-GGUF"
DFLASH27_REV = "2d9571f8ce46e151f61c6499c99dee6079e1d610"
DFLASH27_FILE = ("Qwen3.8-27B-DFlash2-Q4_K_M.gguf", 1143006816,
                 "1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd")
DFLASH27_GB = 1.4   # (its VRAM: weights, context ring, the draft's buffers - 1.33 GB + the pending rows, measured)
TOOL_TEMPLATES = ROOT / "ops" / "abyss-tools"
MANAGED = ("abyss.sh", "common.sh")        # replaced when the template changes (the old one kept as .bak)
ONCE = ("README.md",)                      # written only when missing: the user's own notes go there


def quant_files(quant: str, repo: str = REPO, folder: str | None = None, prefix: str = "") -> list[tuple[str, int, str]]:
    """(path in the repo, size, sha256) of a quant's GGUF files, from Hugging Face's file list (the quant's folder, or
    `folder` with the files whose names start with `prefix`)."""
    try:
        req = urllib.request.Request(HF_TREE.format(repo=repo, folder=quant if folder is None else folder),
                                     headers={"User-Agent": "abyss-setup"})
        items = json.load(urllib.request.urlopen(req, timeout=60))
        files = sorted((i["path"], int(i["size"]), (i.get("lfs") or {}).get("oid", ""))
                       for i in items if i.get("type") == "file" and i["path"].endswith(".gguf")
                       and i["path"].rsplit("/", 1)[-1].startswith(prefix) and "imatrix" not in i["path"])
        if files:
            return files
    except OSError as e:
        if repo != REPO or quant not in KNOWN:
            S.fail(f"cannot read the file list of {repo}/{quant} ({e})", "check the internet connection, or the quant's name")
    if repo == REPO and quant in KNOWN:
        return KNOWN[quant]
    S.fail(f"{repo} has no quant {quant!r}", "names are the folders on https://huggingface.co/" + repo)


def link_mirror(flat: Path, mdir: Path, paths: list[str]) -> bool:
    """When every file is already in `flat`, a plain mirror of the Hugging Face repo (each at its repo path, e.g. a
    quant's file at the repo's top level), link them into `mdir` - setup's quant folder, where the pack and the config
    go - and say True.  Nothing is moved; with any file missing nothing is linked."""
    if not paths or not all((flat / p).is_file() for p in paths):
        return False
    mdir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        dst = mdir / Path(p).name
        if not dst.exists():
            dst.symlink_to((flat / p).resolve())
    return True


def link_file(src: Path, dst: Path) -> str:
    """dst as another name of src: a symlink, else (Windows without the developer mode's symlinks) a hard link - the
    same disk - else a copy.  An existing dst that is src already (or a whole copy of it) is kept; a stale link is
    replaced.  Says how ("symlink", "hard link", "copy", "kept")."""
    if dst.exists():
        if os.path.samefile(src, dst) or (not dst.is_symlink() and dst.stat().st_size == src.stat().st_size):
            return "kept"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        pass
    try:
        os.link(src.resolve(), dst)
        return "hard link"
    except OSError:
        pass
    S.say(f"  (no symlink or hard link here: copying {src.name}, {src.stat().st_size / 1e9:.1f} GB, to {dst.name})")
    shutil.copyfile(src, dst)
    return "copy"


def fetch_files(a, data: Path, groups) -> None:
    """Step 4's downloads: each (repo, local folder, files from quant_files) - a file already there (or in a plain
    mirror of the repo, link_mirror) is not fetched again, the rest is asked once for all (or --download-model); every
    file is checked against Hugging Face's SHA-256 and as a whole GGUF."""
    todo = []
    for repo, mdir, files in groups:
        total = sum(f[1] for f in files)
        S.say(f"  {repo}: {len(files)} file{'s' if len(files) != 1 else ''}, {total / 1e9:.1f} GB -> {mdir}")
        missing = [f for f in files if not (mdir / Path(f[0]).name).exists()]
        if missing and link_mirror(data / "models" / repo, mdir, [f[0] for f in missing]):
            S.ok(f"the files are in {data / 'models' / repo} (a mirror of the repo): linked into {mdir}")
            missing = []
        todo += [(mdir, f) for f in missing]
    if todo:
        where = todo[0][0]
        free = shutil.disk_usage(where.parent if where.parent.exists() else data.parent).free
        S.say(f"  to download: {sum(f[1] for _, f in todo) / 1e9:.1f} GB (free on that disk: {free / 1e9:.0f} GB)")
        if not a.download_model and S.ask("  Download it now?", ["y", "n"], "n", a.yes) != "y":
            S.fail("the model is not downloaded", "run again with --download-model, or put the files into " +
                   " / ".join(str(g[1]) for g in groups))
    for repo, mdir, files in groups:
        for path, size, sha in files:
            dst = mdir / Path(path).name
            S.download(HF_FILE.format(repo=repo, path=path), dst, dst.name)
            if sha and not verified(dst, sha):
                S.fail(f"{dst.name} does not match Hugging Face's SHA-256", "delete it and run again")
        S.check_shards([mdir / Path(f[0]).name for f in files])
    S.ok("every file checked")


def overlay_links(mdir: Path, first: str, base_dir: Path, base_files) -> list[Path]:
    """An overlay GGUF's shards 2..N (MemT-1.0-Flash: shard 1 is the repo's one file, 2-4 are L97's three): linked
    beside shard 1 under its name, <stem>-0000k-of-0000N.gguf -> the base's shard k-1, so iq_pack.py and the engine
    find them by that pattern.  Returns every shard, 1 first."""
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first)
    n = int(m.group(2)) if m else 0
    if not m or int(m.group(1)) != 1 or n != len(base_files) + 1:
        S.fail(f"{first}: not shard 1 of {len(base_files) + 1} (the base has {len(base_files)} shards)",
               "the overlay and its base do not belong together - run setup again after `git pull`")
    stem = first[:m.start()]
    out = [mdir / first]
    how = set()
    for k, (path, _, _) in enumerate(base_files, start=2):
        dst = mdir / f"{stem}-{k:05d}-of-{n:05d}.gguf"
        how.add(link_file(base_dir / Path(path).name, dst))
        out.append(dst)
    S.ok(f"shards 2-{n}: the base's ({', '.join(sorted(how))}) in {mdir}")
    S.check_shards(out)
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verified(path: Path, sha: str) -> bool:
    """The file matches Hugging Face's SHA-256 (a <name>.sha256 mark keeps the answer while the file is unchanged)."""
    mark = path.with_name(path.name + ".sha256")
    st = path.stat()
    key = f"{sha} {st.st_size} {int(st.st_mtime)}"
    if mark.exists() and mark.read_text().strip() == key:
        return True
    S.say(f"  checking {path.name} ({st.st_size / 1e9:.1f} GB) ...")
    if sha256_of(path) != sha:
        return False
    mark.write_text(key)
    return True


def cuda_archs(arch: str | None, gpus) -> str:
    """The CUDA architectures to compile for: --arch, else every chosen GPU's (RTX 50: 120a, DGX Spark's GB10: 121a) -
    "86;120a" for an RTX 3090 beside an RTX 5090."""
    if arch:
        return arch
    out = []
    for g in gpus:
        # RTX 50 (12.0) and GB10 (12.1): the MXFP4 MMQ kernels use Blackwell's FP4 MMA, the arch-specific target's
        cc = {"120": "120a", "121": "121a"}.get(g["arch"], g["arch"])
        if cc not in out:
            out.append(cc)
    return ";".join(sorted(out, key=lambda c: int(c.rstrip("a"))))


def configured_archs() -> str | None:
    cache = ROOT / "build" / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("CMAKE_CUDA_ARCHITECTURES:"):
            return line.split("=", 1)[1].strip()
    return None


def nvcc_major(nvcc: str) -> int:
    """The CUDA toolkit's major version, from `nvcc --version` ("release 12.9, V12.9.86"); 0 when unknown."""
    import re
    import subprocess
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60).stdout
    except OSError:
        return 0
    m = re.search(r"release (\d+)\.", out)
    return int(m.group(1)) if m else 0


def venv_bin(name: str, windows: bool | None = None) -> Path:
    """A program in Abyss's .venv: .venv/Scripts/<name>.exe on Windows (setup-deepseek.bat makes it there),
    .venv/bin/<name> elsewhere (issue #39: the Linux layout was hard-wired)."""
    win = WIN if windows is None else windows
    return ROOT / ".venv" / ("Scripts" if win else "bin") / (name + (".exe" if win else ""))


def build_engine(arch: str | None, gpus, nvcc_arg: str | None = None) -> Path:
    exe = S.engine_path(ROOT / "build")
    cmake = venv_bin("cmake")
    ninja = venv_bin("ninja")
    cc = cuda_archs(arch, gpus)
    nvcc = nvcc_arg or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    conf = [cmake, "-S", ROOT, "-B", ROOT / "build", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cc}",
            "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_COMPILER={nvcc}",
            f"-DSTRATA_GGML_DIR={ROOT / 'third_party' / 'llama.cpp'}"] + S.ggml_libs_flag()
    # (configured again when the GPUs need other architectures - a second card of another generation - or --nvcc names
    # another toolkit than build/ was configured with; then from scratch, S.cmake_fresh)
    if (not (ROOT / "build" / "build.ninja").exists() or configured_archs() not in (None, cc)
            or nvcc_arg and S.cmake_cached(ROOT / "build", "CMAKE_CUDA_COMPILER") not in (None, nvcc_arg)):
        if not Path(nvcc).exists():
            S.fail("nvcc (the CUDA toolkit) was not found", "install the CUDA toolkit (12.8 or newer for RTX 50)")
        # V100 (sm_70): CUDA 13 compiles for Turing (sm_75) and newer only - the V100 needs a CUDA 12 toolkit
        # (12.9 builds sm_70 next to sm_120a, so a V100 beside an RTX 50 builds with it too)
        if any(int(c.rstrip("a")) < 75 for c in cc.split(";")) and nvcc_major(nvcc) >= 13:
            S.fail(f"{nvcc} is CUDA {nvcc_major(nvcc)}, which cannot compile for the V100 (compute capability 7.0)",
                   "install a CUDA 12 toolkit (12.9) beside it and pass --nvcc /usr/local/cuda-12.9/bin/nvcc")
        names = " + ".join(g["name"] for g in gpus)
        S.say(f"  Configuring for {names} (CUDA arch {cc}) ...")
        S.run(conf + S.cmake_fresh(ROOT / "build", conf), quiet=True)
    S.say("  Compiling the engine (a few minutes the first time, only what changed later) ...")
    S.run([cmake, "--build", ROOT / "build", "--target", "strata", "abyss_gpu_tune", "abyss_vision", "-j",
           str(max(2, (os.cpu_count() or 4)))],
          quiet=True)
    S.ok(f"engine: {exe}")
    return exe


def find_nvcc(nvcc_arg: str | None = None) -> str | None:
    """The nvcc build_engine would take, when it is there."""
    c = nvcc_arg or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    return c if Path(c).exists() else None


def use_prebuilt(a, gpus) -> bool:
    """The prebuilt engine (tools/prebuilt.py): --prebuilt, or no nvcc / C++ compiler here.  Checks what it needs -
    an archive for this OS and these GPUs, the NVIDIA driver - and stops with the reason when it cannot be used."""
    compiler = S.find_vcvars() is not None if WIN else shutil.which("g++") is not None
    if not P.want_prebuilt(a.prebuilt, find_nvcc(a.nvcc), compiler):
        return False
    how = "install the CUDA toolkit (12.8 or newer for RTX 50) and a C++ compiler to compile the engine"
    if P.asset_name() is None:
        S.fail("nvcc (the CUDA toolkit) was not found, and no prebuilt engine is made for this OS / CPU "
               "(Windows x64 and Linux x64)", how)
    bad = P.unsupported(g["arch"] for g in gpus)
    if bad:
        S.fail(("" if a.prebuilt else "nvcc (the CUDA toolkit) was not found, and ") + "the prebuilt engine has no code "
               "for compute capability " + ", ".join(f"{x // 10}.{x % 10}" for x in bad) + " (sm_" +
               ", sm_".join(str(x) for x in P.ARCHS) + ": RTX 30 / 40 / 50)", how)
    verdict, msg = P.driver_verdict(P.driver_version())
    if verdict == "stop":
        S.fail(msg)
    (S.warn if verdict == "warn" else S.ok)(msg)
    S.ok("the engine: the prebuilt one from the GitHub release (" + ("--prebuilt" if a.prebuilt else
         "no CUDA toolkit / C++ compiler here") + ")")
    return True


def prebuilt_engine(gpus) -> Path:
    """The prebuilt engine into build/ (abyss.py's install_prebuilt: the same stamp, so ./abyss.sh knows it)."""
    import abyss as M   # noqa: E402  (constants and functions only)
    M.select_build_backend("cuda")
    meta = M.read_json(M.STAMP)
    if M.EXE.exists() and P.current(meta):
        S.ok(f"prebuilt engine {meta['prebuilt']} already here: {M.EXE}")
        return M.EXE
    M.install_prebuilt(S.source_hash(S.ENGINE_SOURCES), [g["index"] for g in gpus])
    return M.EXE


def hip_engine(a):
    """AMD (--backend hip): the GPU check, ROCm and the engine (build-hip/abyss) by abyss.py's HIP helpers (Maya's
    installer: gfx1100, gfx1201, gfx1151; system ROCm 7 on Linux).  Returns (gpus, the config's extra fields)."""
    import argparse as _ap
    sys.path.insert(0, str(ROOT))
    import abyss as M   # noqa: E402  (constants and functions only)
    M.select_build_backend("hip")
    pc = M.check_hip_pc(_ap.Namespace(gpu=a.gpu, gpus=a.gpus if a.gpus != "all" else None, check=False, yes=a.yes))
    llama = Path(S.get_llama_cpp())
    src = S.source_hash(S.ENGINE_SOURCES)
    meta = M.read_json(M.STAMP)
    if not (M.EXE.exists() and meta.get("backend") == "hip" and meta.get("src") == src
            and meta.get("archs") == pc["archs"] and meta.get("rocm") == pc["rocm"]):
        meta = M.compile_engine_hip(pc, llama, src)
    else:
        S.ok(f"HIP engine already compiled: {M.EXE}")
    gpus = pc["gpus"]
    # (as abyss.py's HIP configs: two hardware queues, room for the desktop, one card unless two are chosen, an APU's
    # prompt budget)
    env = {"STRATA_GLM_RAM_HEADROOM_GB": "16", "GPU_MAX_HW_QUEUES": "2"}
    if any(g.get("display", True) for g in gpus):   # (no screen on it - an eGPU beside an APU's desktop: the default)
        env["STRATA_GLM_RESERVE_MB"] = "3072"
    if len(gpus) == 1:
        env["STRATA_GLM_SPLIT"] = "0"
    if any(M.hip_unified_memory(g) for g in gpus):
        env.update({"STRATA_GLM_RESERVE_MB": "1024", "STRATA_GLM_PREFILL_SUB": "1024",
                    "STRATA_GLM_PREFILL_MB": "6144" if S.ram_gb() >= 96 else "4096"})
    elif min(g.get("vram_gb", 0) for g in gpus) >= 20:
        env.update({"STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_PREFILL_MB": "4096"})
    if os.name == "nt" and not any(M.hip_unified_memory(g) for g in gpus):
        # Windows counts the pinned RAM tier against the commit limit (RAM + pagefile) and the 16 GB headroom comes off
        # that: with a small pagefile experts go to the SSD while RAM is free.  The defaults stay safe; this is the
        # tuning that kept every expert resident on an RX 7900 XT + 128 GB (issue #4) - it needs the bigger pagefile.
        S.say("  Windows: if the log's RAM tier holds fewer experts than your RAM allows, a larger pagefile (e.g. 64 GB)")
        S.say("  plus STRATA_GLM_RAM_HEADROOM_GB=4, STRATA_GLM_RESERVE_MB=512 (no display on this card) and --prefill 8192")
        S.say("  in the config kept all experts resident on an RX 7900 XT + 128 GB (github.com/zephel01/Abyss/issues/4)")
    extra = {"exe": str(M.EXE), "backend": "hip", "gpu": [g["index"] for g in gpus],
             "lib_dirs": (meta or {}).get("lib_dirs") or [], "env": env}
    if all(g.get("pci") for g in gpus):   # (Linux) the same cards when HIP numbers them otherwise after a reboot
        extra["gpu_pci"] = [g["pci"] for g in gpus]
    return gpus, extra


# the run config's server settings (serve/server.py reads them; the user sets them by hand): --rewrite-config writes
# the model's part again and keeps these (Maya #1089: a setup run again lost the port, the API key and the host)
SERVER_KEYS = ("port", "host", "api_key", "allow_shutdown", "reasoning_effort", "thinking_budget", "fit_max_tokens",
               "mcp", "mcp_servers", "mcpServers", "parallel")


def write_config(name: str, pack: Path, ctx: int, port: int | None, rewrite: bool, gpus=None,
                 extra: dict | None = None, model: dict | None = None) -> tuple[Path, Path]:
    """port None: the existing config's (--rewrite-config), else 8082; a port given wins."""
    model = model or MODELS["v4"]
    cfg = ROOT / f"abyss-{name}.json"
    run = ROOT / f"run-abyss-{name}.sh"
    if cfg.exists() and not rewrite:
        S.ok(f"{cfg.name} kept (--rewrite-config writes it again)")
        # ... with the libraries of the engine just checked or compiled: after a switch to another ROCm (setup again
        # with ROCM_PATH) the old one's lib first in LD_LIBRARY_PATH would load its runtime into the new engine - ROCm 7
        # and 10 share the sonames (libamdhip64.so.7, libhsa-runtime64.so.1) and the engine has no RUNPATH
        if S.migrate_config_exe(cfg):              # (v1.3.164: build/strata -> build/abyss, once it is there)
            S.ok(f"{cfg.name}: the engine is now {json.loads(cfg.read_text(encoding='utf-8'))['exe']}")
        kept = json.loads(cfg.read_text(encoding="utf-8"))
        if (extra or {}).get("lib_dirs") and kept.get("lib_dirs") != extra["lib_dirs"]:
            kept["lib_dirs"] = extra["lib_dirs"]
            cfg.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
            S.ok(f"{cfg.name}: lib_dirs {', '.join(extra['lib_dirs'])} (the libraries the engine was compiled with)")
    else:
        old = {}
        if cfg.exists():
            try:
                old = json.loads(cfg.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = {}
            old = {k: old[k] for k in SERVER_KEYS if k in old} if isinstance(old, dict) else {}
        port = port if port is not None else old.get("port", 8082)
        cfg.write_text(json.dumps({
            "exe": str(S.engine_path(ROOT / "build")),
            "args": ["--glm-pack", str(pack), "--max-context", str(ctx), "--prefill", "auto"],
            "cwd": str(ROOT),
            "tokenizer": str(pack / "tokenizer"),
            "model_name": model["model_name"],
            # (several: a layer split across them, the first the main card - serve/server.py adds --layer-split auto)
            **({"gpu": [g["index"] for g in gpus]} if gpus else {}),
            "sampling": model.get("sampling", {"temperature": 1.0, "top_p": 0.95}),
            "port": port,
            "log": str(ROOT / f"abyss-{name}.log"),
            **(extra or {}),
            **{k: v for k, v in old.items() if k != "port"},
        }, indent=1) + "\n", encoding="utf-8")
        kept = [k for k in old if k != "port" or old[k] == port]
        S.ok(f"{cfg.name} written (context {ctx}, port {port})" +
             (f" - kept from the old one: {', '.join(kept)}" if kept else ""))
    py = venv_bin("python")
    run.write_text("#!/bin/sh\n"
                   f"# {model['title']} ({name}) on Abyss: the dashboard and API (written by setup-deepseek.sh)\n"
                   f'cd "{ROOT}" || exit 1\n'
                   f'exec "{py}" "{ROOT / "serve" / "server.py"}" --engine strata "--config" "{cfg}" '
                   f'"--port" "{json.loads(cfg.read_text())["port"]}" "$@"\n', encoding="utf-8")
    run.chmod(0o755)
    S.ok(f"{run.name} written")
    return cfg, run


def dspark_step(a, data: Path, cfg: Path, fresh: bool, had: str | None, gpus, backend: str) -> None:
    """The DSpark drafter: --dspark downloads it (SHA-256 checked) and names it in the config's env; --no-dspark takes
    it out.  A new config asks (default no); a config that has it keeps it - also through --rewrite-config."""
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    env = conf.setdefault("env", {})
    have = env.get("STRATA_DS4_DSPARK") or None

    def save(what: str) -> None:
        cfg.write_text(json.dumps(conf, indent=1) + "\n", encoding="utf-8")
        S.ok(what)

    if a.no_dspark:
        if have:
            del env["STRATA_DS4_DSPARK"]
            save(f"DSpark drafter off in {cfg.name} (the file stays: {have})")
        else:
            S.ok("DSpark drafter: off")
        return
    if len(gpus) > 1:   # (the engine runs the drafter on one device only)
        S.ok("DSpark drafter: not with a layer split" + (" - left out of the config" if have else ""))
        if have:
            del env["STRATA_DS4_DSPARK"]
            save(f"{cfg.name} written without STRATA_DS4_DSPARK")
        return
    if have and not a.dspark:
        S.ok(f"DSpark drafter: on ({have})")
        return
    if had and not a.dspark and Path(had).exists():   # (--rewrite-config: the old config's drafter)
        env["STRATA_DS4_DSPARK"] = had
        save(f"DSpark drafter: on, as before ({had})")
        return
    name, size, sha = DSPARK_FILE
    dst = data / "models" / DSPARK_REPO / name
    if not a.dspark:
        if not fresh:
            S.ok("DSpark drafter: off (./setup-deepseek.sh --dspark downloads it, 6.9 GB, and turns it on)")
            return
        S.say("  DeepSeek's DSpark drafter (6.9 GB) proposes up to 5 tokens a step, and the model checks them in one pass")
        S.say("  (the answers are still the model's own): on an AI PRO R9700 + 94 GB RAM, swe-bench coding tasks ran at")
        S.say("  61.9 tok/s against 48.4. It takes ~6.5 GB of VRAM from the experts" +
              (" (measured on AMD; the NVIDIA build compiles, not yet measured)" if backend == "cuda" else ""))
        if min((g.get("vram_gb") or 0) for g in gpus) < 30:
            S.say("  (measured on a 32 GB card; on a smaller one it takes a larger share of the experts' room)")
        if S.ask("  Download it and turn it on?", ["y", "n"], "n", a.yes) != "y":
            S.ok("DSpark drafter: off (./setup-deepseek.sh --dspark turns it on later)")
            return
    S.say(f"  {name}, {size / 1e9:.2f} GB -> {dst.parent}")
    S.download(DSPARK_URL.format(repo=DSPARK_REPO, rev=DSPARK_REV, file=name), dst, name)
    if not verified(dst, sha):
        S.fail(f"{dst.name} does not match Hugging Face's SHA-256", "delete it and run again")
    env["STRATA_DS4_DSPARK"] = str(dst)
    save(f"DSpark drafter on in {cfg.name} (STRATA_DS4_DSPARK)")


# Inkling 2 Mini's DSpark drafter (v1.3.192): unsloth's dspark-Inkling-2-Mini-Q8_0.gguf (the repo's dspark/ folder,
# 0.6 GB in VRAM; src/core/ink_dspark.cu, CUDA, one GPU).  Setup does not download it.  The engine finds it itself
# beside the pack (inkd_find: the hf layout next to the model's folder, or the folder itself) and keeps it when the
# expert pool comes out the same with it; the config's env names it only by hand (STRATA_INK_DSPARK=<file>: loaded
# whatever the VRAM) or turns it off (=0, --no-dspark).  Since 2026-11-23 unsloth's file under INK_DSPARK_NAME is a
# newer dflash layout (markov_w2, no dspark_fuse) the engine does not read yet; the layout it reads is kept there as
# INK_DSPARK_OLD (the repo's top folder), which the engine prefers when both are in one place
INK_DSPARK_NAME = "dspark-Inkling-2-Mini-Q8_0.gguf"
INK_DSPARK_OLD = "dspark-Inkling-2-Mini-Q8_0_Old.gguf"


def ink_dspark_found(mdir: Path) -> Path | None:
    """Where the engine finds the drafter for a pack in mdir (the same places and order, ink_dspark.cu's inkd_find)."""
    ud = mdir.parents[2] / "unsloth" / "Inkling-2-Mini-GGUF"
    return next((d / n for d in (mdir.parent / "dspark", mdir.parent, mdir, ud / "dspark", ud)
                 for n in (INK_DSPARK_OLD, INK_DSPARK_NAME) if (d / n).is_file()), None)


def ink_dspark_new_layout(p: Path) -> bool:
    """True for unsloth's newer dflash drafter (2026-11-23: markov_w2, no dspark_fuse), which the engine skips."""
    try:
        import gguf_io
        names = {t.name for t in gguf_io.Reader(p, data=False).tensors}
    except Exception:
        return False
    return "markov_w2.weight" in names and "dspark_fuse.weight" not in names


def ink_dspark_step(a, mdir: Path, cfg: Path, gpus, backend: str) -> None:
    """Inkling 2 Mini's DSpark drafter: on by default when its file is on disk where the engine finds it (mdir/../dspark/,
    mdir/.., mdir, unsloth's folder beside an abliterated model's) - the engine keeps it when the expert pool holds as
    many experts with it as without.  --no-dspark writes STRATA_INK_DSPARK=0 (the file stays).  A config that names
    the file setup found (v1.3.195-v1.3.199 wrote it: loaded whatever the VRAM) loses the line, so the engine's check
    applies; one that names another file keeps it.  CUDA and one GPU only (the engine's drafter runs there)."""
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    env = conf.setdefault("env", {})
    have = env.get("STRATA_INK_DSPARK") or None

    def save(what: str) -> None:
        cfg.write_text(json.dumps(conf, indent=1) + "\n", encoding="utf-8")
        S.ok(what)

    found = ink_dspark_found(mdir)
    if a.no_dspark:
        if have != "0":
            env["STRATA_INK_DSPARK"] = "0"
            save(f"DSpark drafter off in {cfg.name} (STRATA_INK_DSPARK=0; the file stays" +
                 (f": {have})" if have else ")"))
        else:
            S.ok("DSpark drafter: off (--no-dspark)")
        return
    if backend != "cuda" or len(gpus) > 1:
        if have and have != "0":
            del env["STRATA_INK_DSPARK"]
            save(f"DSpark drafter: NVIDIA and one GPU only - {cfg.name} written without STRATA_INK_DSPARK")
        else:
            S.ok("DSpark drafter: NVIDIA and one GPU only")
        return
    if have == "0":
        S.ok(f"DSpark drafter: off ({cfg.name} sets STRATA_INK_DSPARK=0; take the line out to turn it on)")
        return
    if have and (found is None or Path(have) != found):
        S.ok(f"DSpark drafter: on ({have}, named in {cfg.name}: loaded whatever the VRAM)")
        return
    if have:
        del env["STRATA_INK_DSPARK"]
        save(f"DSpark drafter: the engine finds {found} itself - {cfg.name} no longer names it, so it is kept only "
             "when the experts fit beside it")
        return
    if found is None:
        S.ok(f"DSpark drafter: off - put unsloth's {INK_DSPARK_OLD} (the repo's top folder) in "
             f"{mdir.parent / 'dspark'} and the engine loads it")
        return
    if ink_dspark_new_layout(found):
        S.warn(f"DSpark drafter: {found} is the newer drafter layout (unsloth's file under this name since "
               f"2026-11-23), which Abyss does not read yet - the engine runs without a drafter; put unsloth's "
               f"{INK_DSPARK_OLD} beside it and the engine loads that one")
        return
    S.ok(f"DSpark drafter: on ({found}; the engine keeps it when the experts fit beside it, --no-dspark turns it off)")


def dflash_step(a, data: Path, cfg: Path, gpus) -> None:
    """Qwen3.8's DFlash 2 drafter, opt-in: --dflash downloads it (SHA-256 checked) and names it in the config's env;
    --no-dflash takes it out.  Without either the config stays as it is."""
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    env = conf.setdefault("env", {})
    have = env.get("STRATA_Q4X_DFLASH") or None

    def save(what: str) -> None:
        cfg.write_text(json.dumps(conf, indent=1) + "\n", encoding="utf-8")
        S.ok(what)

    if a.no_dflash or (have and len(gpus) > 1):   # (the engine runs it on one device only)
        if have:
            del env["STRATA_Q4X_DFLASH"]
            save(f"DFlash 2 drafter off in {cfg.name} (the file stays: {have})")
        return
    if not a.dflash:
        if have:
            S.ok(f"DFlash 2 drafter: on ({have})")
        return
    if len(gpus) > 1:
        S.ok("DFlash 2 drafter: not with a layer split")
        return
    name, size, sha = DFLASH_FILE
    dst = data / "models" / DFLASH_REPO / name
    S.say(f"  {name}, {size / 1e9:.2f} GB -> {dst.parent}")
    S.download(DSPARK_URL.format(repo=DFLASH_REPO, rev=DFLASH_REV, file=name), dst, name)
    if not verified(dst, sha):
        S.fail(f"{dst.name} does not match Hugging Face's SHA-256", "delete it and run again")
    env["STRATA_Q4X_DFLASH"] = str(dst)
    save(f"DFlash 2 drafter on in {cfg.name} (STRATA_Q4X_DFLASH)")


def dflash27_step(a, data: Path, cfg: Path, gpus, fits: bool) -> None:
    """Qwen3.8-27B's DFlash 2 drafter, on by default where the whole file and the drafter fit in one GPU's VRAM: the
    file downloaded (SHA-256 checked) and named in the config's env; --no-dflash takes it out (the file stays),
    --dflash sets it up on a smaller GPU too (the engine drops it there unless STRATA_Q35_DFLASH_KEEP=1)."""
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    env = conf.setdefault("env", {})
    have = env.get("STRATA_Q35_DFLASH") or None

    def save(what: str) -> None:
        cfg.write_text(json.dumps(conf, indent=1) + "\n", encoding="utf-8")
        S.ok(what)

    if a.no_dflash or (have and len(gpus) > 1):   # (the engine runs it on one device only)
        if have:
            del env["STRATA_Q35_DFLASH"]
            save(f"DFlash 2 drafter off in {cfg.name} (the file stays: {have})")
        else:
            S.ok("DFlash 2 drafter: off (--no-dflash)")
        return
    if len(gpus) > 1:
        S.ok("DFlash 2 drafter: not with a layer split")
        return
    if have and Path(have).exists():
        S.ok(f"DFlash 2 drafter: on ({have})")
        return
    if not a.dflash and not fits:
        S.ok(f"DFlash 2 drafter: off - the model and the drafter (~{DFLASH27_GB} GB) do not both fit in this GPU's VRAM "
             "(--dflash sets it up anyway; the engine then drops it while the model's FFN does not fit beside it)")
        return
    name, size, sha = DFLASH27_FILE
    dst = data / "models" / DFLASH27_REPO / name
    S.say("  z-lab's DFlash 2 drafter proposes up to 7 tokens a step beside the NextN block (the answers are still the")
    S.say("  model's own): code +15-21%, reasoning +13-16% on an RTX 5090; ~1.4 GB of VRAM (--no-dflash: off)")
    S.say(f"  {name}, {size / 1e9:.2f} GB -> {dst.parent}")
    S.download(DSPARK_URL.format(repo=DFLASH27_REPO, rev=DFLASH27_REV, file=name), dst, name)
    if not verified(dst, sha):
        S.fail(f"{dst.name} does not match Hugging Face's SHA-256", "delete it and run again")
    env["STRATA_Q35_DFLASH"] = str(dst)
    save(f"DFlash 2 drafter on in {cfg.name} (STRATA_Q35_DFLASH)")


def make_tools(tools: Path) -> None:
    """The Abyss-tools folder from ops/abyss-tools: abyss.sh and common.sh follow the template (a changed one is kept
    as .bak), README.md is written once."""
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "logs").mkdir(exist_ok=True)
    for name in MANAGED + ONCE:
        text = (TOOL_TEMPLATES / name).read_text(encoding="utf-8").replace("@ABYSS@", str(ROOT))
        dst = tools / name
        if dst.exists():
            old = dst.read_text(encoding="utf-8")
            if old == text:
                continue
            if name in ONCE:
                S.ok(f"{dst} kept (yours)")
                continue
            dst.with_name(name + ".bak").write_text(old, encoding="utf-8")
            S.warn(f"{dst} updated from the template (the previous one: {name}.bak)")
        dst.write_text(text, encoding="utf-8")
        if name.endswith(".sh"):
            dst.chmod(0o755)
    S.ok(f"tools: {tools}  ({tools / 'abyss.sh'} start | stop | status | update | log | models)")


def _shared_memory(g) -> bool:
    """A GPU without memory of its own: an APU (Ryzen AI Max's Radeon 8060S, gfx1151) or DGX Spark's GB10 - its VRAM
    is the system RAM (as abyss.py's hip_unified_memory, plus setup.py's "integrated")."""
    return g.get("arch") == "gfx1151" or bool(g.get("integrated")) or bool(
        re.search(r"Ryzen AI Max|Radeon(?:\s*\(TM\))?\s+80[4-6]\dS", g.get("name", ""), re.I))


def pc_memory(gpus, ram: float) -> tuple[float, float, bool]:
    """(VRAM, RAM, shared) in GB for picking a PC-size file: the chosen GPUs' VRAM (a layer split's cards add up), the
    RAM.  An APU's memory is counted once, as RAM (shared=True when there is no GPU memory of its own): an AMD APU's
    BIOS carve-out (sysfs / the registry's "VRAM") is not in the OS's RAM, so it is added to it; DGX Spark's "VRAM" is
    the RAM itself."""
    vram, shared = 0.0, False
    for g in gpus or []:
        if _shared_memory(g):
            shared = True
            if g.get("vendor") == "amd" and (g.get("driver") == "amdgpu" or g.get("source") == "registry"):
                ram += g.get("vram_gb") or 0.0
        else:
            vram += g.get("vram_gb") or 0.0
    return vram, ram, shared and vram == 0


def fit_quant(key: str, vram: float, ram: float, shared: bool = False) -> dict | None:
    """The PC-size file ("variants") that suits a PC best: the largest one whose GPU size is at most this VRAM (+1 GB:
    a 16 GB card reports 15.9) and whose RAM size at most this RAM (+2 GB, or 4% - a 128 GB PC with an iGPU shows ~123),
    among the default model's files (Ornith: the 122B; the 397B ones are "extra", the largest that fits).  An APU
    (shared) holds both parts in one memory: GPU + RAM size at most its RAM.  When none fits, the smallest
    (fits=False).  None for a model without PC-size files."""
    model = MODELS[key]
    variants = model.get("variants") or {}
    if not variants:
        return None

    def size(q):
        v = variants[q]
        return (v.get("gb", model.get("gb", 0)), v["vram"] + v["ram"])

    def fits(q):
        v = variants[q]
        slack = max(2.0, 0.04 * v["ram"])
        if shared:
            return v["vram"] + v["ram"] <= ram + slack + 1
        return v["vram"] <= vram + 1 and v["ram"] <= ram + slack

    base = model["model_name"]
    family = sorted((q for q in variants if variants[q].get("model_name", base) == base), key=size)
    ok = [q for q in family if fits(q)]
    pick = ok[-1] if ok else family[0]
    extra = [q for q in sorted(variants, key=size) if q not in family and fits(q)]
    return {"quant": pick, "fits": bool(ok), "others": [q for q in family if q != pick],
            "extra": extra[-1] if extra else None}


def dense_shift(key: str, quant: str | None) -> tuple[float, float]:
    """Another file of a model that runs all in VRAM: (GB more than the default file the tables were measured with,
    its size relative to that file) - (0, 1) for the default or an unknown name."""
    m = MODELS[key]
    q = m.get("quant_gb") or {}
    if quant is None or quant not in q or m["quant"] not in q:
        return 0.0, 1.0
    return q[quant][0] - q[m["quant"]][0], q[quant][0] / q[m["quant"]][0]


def dense_need(key: str, ctx: int, quant: str | None = None) -> float | None:
    """A model that runs all in VRAM ("vram_ctx"): the GPU size (GB) a context needs - the measured context at or above
    it; None above the longest measured."""
    need = MODELS[key]["vram_ctx"]
    at = [c for c in need if c >= ctx]
    return need[min(at)] + dense_shift(key, quant)[0] if at else None


def dense_context(key: str, vram: float, ctx: int, quant: str | None = None) -> int | None:
    """The context to write for a model that runs all in VRAM: `ctx` when this GPU holds it, else the longest measured
    one below it that it holds; None when it holds none (+0.5 GB: a 24 GB card reports 23.6-24.0)."""
    n = dense_need(key, ctx, quant)
    if n is not None and vram + 0.5 >= n:
        return ctx
    shift = dense_shift(key, quant)[0]
    fits = [c for c, g in MODELS[key]["vram_ctx"].items() if vram + 0.5 >= g + shift and c < ctx]
    return max(fits) if fits else None


def dense_split_context(key: str, vram: float, ctx: int, quant: str | None = None) -> int | None:
    """dense_context for a GPU smaller than the file: the FFN rows that do not fit stay in RAM ("ffn_gb" less VRAM)."""
    m = MODELS[key]
    return dense_context(key, vram + m.get("ffn_gb", 0.0) * dense_shift(key, quant)[1], ctx, quant) \
        if m.get("ffn_gb") else None


def dense_pick(key: str, vram: float) -> str | None:
    """The file of a model that runs all in VRAM ("picks") this GPU suits: the largest that holds an 8K context all in
    VRAM (dense_gpu_check then gives it the longest context it holds); None when none does (the default file, its FFN
    partly in RAM)."""
    ok = [q for q in MODELS[key].get("picks") or () if dense_context(key, vram, 8192, q) == 8192]
    return ok[-1] if ok else None


def dense_gpu_check(a, key: str, gpus, backend: str, shared: bool, ram: float | None = None,
                    quant: str | None = None) -> int:
    """Step 1 for a dense model ("vram_ctx"): one GPU that holds the file, the KV cache and the prompt path - or, on a
    smaller GPU, all but the FFN rows the engine keeps in RAM (slower).  Returns the context to write (shorter than
    --context when this GPU holds only a shorter one); stops on a GPU too small for the shortest, or too little RAM
    for the rows it leaves there, before anything is downloaded or compiled."""
    model = MODELS[key]
    need = model["vram_ctx"]
    if len(gpus) > 1:
        S.fail(f"{model['title']} runs on one GPU in this version (no layer split across GPUs yet)",
               "set it up without --gpus (--gpu picks the card)")
    if backend == "hip":
        S.warn(f"{model['title']} is measured on NVIDIA only so far (RTX 5090): on AMD it is untested")
    if shared:
        S.say(f"  ({model['title']} on a GPU that shares the RAM: untested so far; the whole file goes into that memory)")
        return a.context
    vram = gpus[0].get("vram_gb") or 0.0
    shift, scale = dense_shift(key, quant)
    quant = quant or model["quant"]
    ctx = dense_context(key, vram, a.context, quant)
    table = ", ".join(f"{c // 1024}K: {g + shift:.0f} GB" for c, g in sorted(need.items()))
    if ctx == a.context:
        S.ok(f"{model['title']}: the whole file in VRAM - this {vram:.0f} GB GPU holds a {ctx // 1024}K context "
             f"(the GPU size a context needs: {table})")
        return ctx
    # short of that: the FFN rows that do not fit in RAM, on the CPU (the context asked for, or the longest it holds)
    sctx = dense_split_context(key, vram, a.context, quant)
    small = model.get("small_gpu")
    ffn_gb = model.get("ffn_gb", 0.0) * scale
    # (an 8 GB card holds the rest of the default file; the rest of a larger one needs that much more)
    if sctx is None and ctx is None and small and vram >= small[0] + max(0.0, shift - (scale - 1) * model.get("ffn_gb", 0.0)):
        sctx = small[1]   # (an 8 GB card: 8K, the head partly on the CPU too)
    if sctx is not None and (ctx is None or sctx > ctx):
        ram = S.ram_gb() if ram is None else ram
        if ram + 0.5 < model.get("ram_min", 0):
            S.fail(f"{model['title']} on this {vram:.0f} GB GPU keeps part of its FFN in RAM (up to "
                   f"~{ffn_gb:.0f} GB, pinned): it needs {model['ram_min']} GB of RAM, and this PC has {ram:.0f} GB",
                   f"a PC with {model['ram_min']} GB of RAM or more, or a GPU that holds the file (24 GB or more)")
        S.ok(f"{model['title']}: this {vram:.0f} GB GPU holds a {sctx // 1024}K context with part of the FFN in RAM "
             + (f"(the engine computes those rows on the CPU: slower than all in VRAM - RTX 5090 capped to 12 / 16 GB: "
                f"~16 / ~27 / ~65 tok/s with the drafter at 8 / 12 / 16 GB, 20 GB and up all in VRAM ~130)"
                if key == "qwen38-27b" else "(the engine computes those rows on the CPU: slower than all in VRAM)"))
        if small and sctx == small[1] and vram < 9.0:
            S.warn(f"an {vram:.0f} GB GPU runs {model['title']} with ~0.2 GB to spare at 8K: keep other programs "
                   "(a browser, the desktop on another GPU if you can) off it")
        if sctx < a.context:
            S.warn(f"--context {a.context} is longer than this GPU holds: the config gets {sctx} (a larger GPU holds more)")
        return sctx
    if ctx is None:
        gb = (model.get("quant_gb") or {}).get(quant, (model["gb"],))[0]
        least = model.get("gpu_min", 8)
        S.fail(f"{model['title']} needs a GPU with {least} GB or more: the {quant} file (~{gb:.0f} GB) runs "
               + (f"from VRAM, its FFN partly from RAM, and this GPU has {vram:.0f} GB (the GPU size a context needs "
                  f"all in VRAM: {table}; ~{ffn_gb:.0f} GB less with the FFN in RAM)" if model.get("ffn_gb") else
                  f"from VRAM, and this GPU has {vram:.0f} GB (the GPU size a context needs: {table}; less is not "
                  "measured)"),
               f"a GPU with {least} GB or more; on a smaller one, Qwen3.8-Flash-Next (--model qwen38) or "
               "Ornith-1.5-35B-A3B (--model ornith35) run from VRAM + RAM")
    S.ok(f"{model['title']}: the whole file in VRAM - this {vram:.0f} GB GPU holds a {ctx // 1024}K context "
         f"(the GPU size a context needs: {table})")
    n = dense_need(key, a.context, quant)
    S.warn(f"--context {a.context} " + (f"needs a ~{n:.0f} GB GPU" if n else "is longer than measured") +
           f": the config gets {ctx} (--context {ctx} says so; a larger GPU holds more)")
    return ctx


def pc_line(vram: float, ram: float, shared: bool) -> str:
    if shared:
        return f"this PC: {ram:.0f} GB shared memory (an APU: GPU and RAM in one)"
    return f"this PC: {vram:.0f} GB GPU + {ram:.0f} GB RAM" if vram else f"this PC: no GPU memory + {ram:.0f} GB RAM"


def choose_quant(a, key: str, gpus) -> str:
    """--quant when given; else the PC-size file for this PC (fit_quant), asked with it as the default (--yes takes it).
    DeepSeek V4 keeps unsloth's UD-Q2_K_XL (the default since v1.3.0) and only names zephel01's file that suits the PC;
    a file of this model already set up (its config there, --name not given) is kept and the better one named."""
    model = MODELS[key]
    if a.quant:
        if model.get("quant_gb") and a.quant not in model["quant_gb"]:
            S.fail(f"{model['title']} has no file {a.quant!r}", "one of: " + ", ".join(model["quant_gb"]))
        return a.quant
    vram, ram, shared = pc_memory(gpus, S.ram_gb())
    if model.get("picks"):
        # (a model all in VRAM: the default stays the tested file - a setup run again must not start another download -
        # and the file this GPU holds best is named)
        pick = dense_pick(key, vram) if not shared else None
        if pick and pick != model["quant"]:
            gb = model["quant_gb"][pick][0]
            S.say(f"  {pc_line(vram, ram, shared)} -> --quant {pick} (~{gb:.0f} GB) is the largest of "
                  f"{', '.join(model['picks'])} this GPU holds all in VRAM; the default stays {model['quant']} "
                  f"(every file of {model['repo']} runs: --quant names it)")
        return S.ask(f"  Quant ({model['quant']} or {pick})" if pick and pick != model["quant"] else
                     f"  Quant ({model['quant']})", [model["quant"], *([pick] if pick and pick != model["quant"] else [])],
                     model["quant"], a.yes)
    rec = fit_quant(key, vram, ram, shared)
    if rec is None:
        return model["quant"]
    variants = model["variants"]
    v = variants[rec["quant"]]
    why = "" if rec["fits"] else (f"  (less than every file here is made for: {rec['quant']}, the smallest, is for a "
                                  f"{v['vram']} GB GPU + {v['ram']} GB RAM - it runs, with more of it read from the SSD: "
                                  "slower)")
    note = ""
    if rec["extra"]:
        x = variants[rec["extra"]]
        note = (f"  ({x.get('title', model['title'])} fits this PC too: --quant {rec['extra']} - a bigger model, "
                "slower: more of it is read from the SSD)")
    others = f" (the others: {', '.join(rec['others'])})" if rec["others"] else ""
    if key == "v4":
        # (the default stays: a setup run again after `git pull` must not start another ~60-100 GB download)
        S.say(f"  {pc_line(vram, ram, shared)} -> zephel01's --quant {rec['quant']} is the file made for this size"
              f"{others}; the default stays {model['quant']} (unsloth's, the tested one)")
        if why:
            S.say(why)
        return S.ask(f"  Quant ({model['quant']} or {', '.join(variants)})", [model["quant"], *variants],
                     model["quant"], a.yes)
    have = [] if a.name else [q for q in variants if (ROOT / f"abyss-{model['prefix']}-{q.lower()}.json").exists()]
    default = rec["quant"]
    if have and default not in have:
        # (set up before: kept, so a setup run again after `git pull` does not start another download)
        default = model["quant"] if model["quant"] in have else have[-1]
        S.say(f"  {pc_line(vram, ram, shared)} -> --quant {rec['quant']}{others}; already set up here: {default} "
              f"(kept; --quant {rec['quant']} sets up the other)")
    else:
        S.say(f"  {pc_line(vram, ram, shared)} -> --quant {rec['quant']}{others}")
        if why:
            S.say(why)
    if note:
        S.say(note)
    return S.ask(f"  Quant ({', '.join(variants)})", list(variants), default, a.yes)


def side_file(a, repo: str, path: str, mdir: Path, what: str) -> Path:
    """A file of the model's repo beside the GGUF (unsloth's MTP/ block, the mmproj): size and SHA-256 from Hugging
    Face's file list, checked after the download like the model's files.  Returns where it is."""
    folder, _, fname = path.rpartition("/")
    got = [f for f in quant_files(folder or "", repo, folder, fname) if f[0] == path]
    if not got:
        S.fail(f"{repo} has no {path}", "check the repo on https://huggingface.co/" + repo)
    _, size, sha = got[0]
    dst = mdir / path
    if not dst.exists() or dst.stat().st_size != size:
        S.say(f"  {what}: {path}, {size / 1e9:.2f} GB")
        if not a.download_model and S.ask("  Download it now?", ["y", "n"], "n", a.yes) != "y":
            S.fail(f"{path} is not downloaded", "run again with --download-model, or put it into " + str(dst.parent))
        dst.parent.mkdir(parents=True, exist_ok=True)
        S.download(HF_FILE.format(repo=repo, path=path), dst, dst.name)
    if dst.stat().st_size != size or (sha and not verified(dst, sha)):
        S.fail(f"{dst.name} does not match Hugging Face's size / SHA-256", "delete it and run again")
    S.ok(f"{path} checked")
    return dst


def borrowed_mtp(a, repo: str, fname: str, mdir: Path, title: str) -> Path | None:
    """A draft block from another repo (MiMo-9B borrows Qwen3.5-9B's NextN block): into MTP/ beside the GGUF, where the
    engine takes it by itself (size and SHA-256 from Hugging Face's file list, checked like the model's files).  None,
    with a note, when the repo does not have it or cannot be read: the model then runs without it (n-gram drafts)."""
    dst = mdir / "MTP" / fname
    try:
        req = urllib.request.Request(HF_TREE.format(repo=repo, folder=""), headers={"User-Agent": "abyss-setup"})
        items = json.load(urllib.request.urlopen(req, timeout=60))
        got = [(int(i["size"]), (i.get("lfs") or {}).get("oid", "")) for i in items
               if i.get("type") == "file" and i.get("path") == fname]
    except (OSError, ValueError):
        got = []
    if not got:
        if dst.exists():
            S.ok(f"draft block: {dst} (already here; {repo} cannot be read now)")
            return dst
        S.ok(f"draft block: not available ({repo}/{fname} cannot be read) - {title} runs without it "
             "(n-gram drafts only: ~20-25% slower decode); setup again fetches it once it is there")
        return None
    size, sha = got[0]
    if not dst.exists() or dst.stat().st_size != size:
        S.say(f"  the draft block: {repo}/{fname}, {size / 1e9:.2f} GB")
        if not a.download_model and S.ask("  Download it now?", ["y", "n"], "y", a.yes) != "y":
            S.ok(f"draft block: not downloaded - {title} runs without it (setup again asks)")
            return None
        dst.parent.mkdir(parents=True, exist_ok=True)
        S.download(HF_FILE.format(repo=repo, path=fname), dst, dst.name)
    if dst.stat().st_size != size or (sha and not verified(dst, sha)):
        S.fail(f"{dst.name} does not match Hugging Face's size / SHA-256", "delete it and run again")
    S.ok(f"draft block: MTP/{fname} checked")
    return dst


def vision_27b_step(a, repo: str, mdir: Path, gguf: Path, cfg: Path, gpus, backend: str, resident: bool = False) -> None:
    """Qwen3.8-27B reads images: unsloth's mmproj-F16.gguf (0.93 GB) and strata-vision (llama.cpp's image library,
    built once like ./abyss.sh's for GLM-5.3-Flash) - asked, default yes; the config's "vision" entry names them (the
    encoder reads only the vocabulary of the model's GGUF).  --no-vision, or an AMD GPU (text only for now), skips it."""
    S.step(9, "images (the vision encoder)")
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    if getattr(a, "no_vision", False):
        S.ok("skipped (--no-vision): the model reads text only")
        return
    if backend != "cuda":
        S.ok("images: NVIDIA only for now - the model reads text only")
        return
    import abyss as M   # noqa: E402  (its vision build: llama.cpp's mtmd with tools/vision)
    if getattr(a, "_prebuilt", False) and not M.VEXE.exists():
        S.ok("images: not with the prebuilt engine yet (the vision encoder is compiled: the CUDA toolkit) - the model "
             "reads text only")
        return
    if conf.get("vision") and Path(conf["vision"].get("exe", "")).exists() and Path(conf["vision"].get("mmproj", "")).exists():
        S.ok(f"images: {conf['vision']['mmproj']} (already set up)")
        return
    if S.ask("  Set up images (the vision encoder: a 0.93 GB download and a 10-20 minute build, once)?", ["y", "n"], "y",
             a.yes) != "y":
        S.ok("images: not set up - the model reads text only (setup again asks)")
        return
    mmproj = side_file(a, repo, DENSE27_MMPROJ, mdir, "the image encoder's weights")
    nvcc = a.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    pc = {"archs": cuda_archs(a.arch, gpus).split(";"), "nvcc": nvcc, "backend": backend}
    src = S.source_hash(("tools/vision",))
    vmeta = M.read_json(M.VSTAMP)
    if not (M.VEXE.exists() and vmeta.get("src") == src and set(pc["archs"]) <= set(vmeta.get("archs", []))):
        if not M.compile_vision(pc, {}, S.get_llama_cpp(S.llama_cpp_reasons(gpu_engine=False, images=True)), src):
            S.warn("the vision encoder did not compile (the reason is above): the model reads text only - setup again "
                   "tries again")
            return
    # (the dense model has no expert cache to lend the encoder: on a GPU with ~1.5 GB to spare beside the model and its
    # context the encoder stays resident there (~1 GB), else it runs on the CPU)
    conf["vision"] = {"exe": str(M.VEXE), "mmproj": str(mmproj), "model": str(gguf),
                      **({"gpu": True, "resident": True} if resident else {"gpu": False})}
    cfg.write_text(json.dumps(conf, indent=1) + "\n", encoding="utf-8")
    S.ok(f"images: {M.VEXE} with {mmproj.name} ({cfg.name})")


def vision_ling_step(a, repo: str, mdir: Path, gguf: Path, cfg: Path, backend: str) -> None:
    """Ling-3.2-air reads images: bartowski's mmproj-Ling-3.2-air-f16.gguf (0.87 GB) and abyss-vision, Abyss's own
    encoder (src/vision/abyss_vision.cu, compiled with the engine - no llama.cpp) - asked, default yes.  The encoder
    starts while a request's pictures are encoded, in GPU memory the model lends it (the expert cache's tail), as
    GLM-5.3-Flash's does.  --no-vision, an AMD GPU (untested) or a prebuilt engine without abyss-vision skip it."""
    S.step(9, "images (the vision encoder)")
    conf = json.loads(cfg.read_text(encoding="utf-8"))
    if getattr(a, "no_vision", False):
        S.ok("skipped (--no-vision): the model reads text only")
        return
    if backend != "cuda":
        S.ok("images: NVIDIA only for now - the model reads text only")
        return
    exe = Path(conf["exe"]).with_name("abyss-vision" + (".exe" if WIN else ""))
    if not exe.exists():
        S.ok("images: this engine has no abyss-vision (the prebuilt engine does not carry it yet) - the model reads "
             "text only")
        return
    if conf.get("vision") and Path(conf["vision"].get("mmproj", "")).exists():
        S.ok(f"images: {conf['vision']['mmproj']} (already set up)")
        return
    if S.ask("  Set up images (the vision encoder's weights: a 0.87 GB download, once)?", ["y", "n"], "y", a.yes) != "y":
        S.ok("images: not set up - the model reads text only (setup again asks)")
        return
    mmproj = side_file(a, repo, LING32_MMPROJ, mdir, "the image encoder's weights")
    conf["vision"] = {"exe": str(exe), "mmproj": str(mmproj), "model": str(gguf), "gpu": True, "prep": "ling3vl"}
    cfg.write_text(json.dumps(conf, indent=1) + "\n", encoding="utf-8")
    S.ok(f"images: {exe.name} with {mmproj.name} ({cfg.name})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up DeepSeek V4 Flash / V4.1 Flash 1017 on Abyss")
    ap.add_argument("--model", choices=tuple(MODELS), default="v4", help="v4: DeepSeek V4 Flash 0731 (unsloth's GGUF, "
                    "the default); v41: DeepSeek V4.1 Flash 1017 (zephel01's LDLQ GGUF for a 32 GB GPU + 96 GB RAM, BETA); "
                    "ornith: Ornith-1.5-122B, ornith35: Ornith-1.5-35B-A3B (the official Q4_K_M, for an 8-16 GB GPU), laguna: Laguna-S-2.1 (unsloth's UD-IQ4_XS, a 16-32 GB GPU + 64 GB RAM), inkling2mini: Inkling 2 Mini (unsloth's UD-Q4_K_XL, a 12-32 GB GPU + 32 GB RAM, text only), inkling2mini-abl: the same abliterated (orcarouter's Mix-16G, a 16-24 GB GPU), ling32air: Ling-3.2-air (bartowski's Q4_K_M, a 24-32 GB GPU + 64-96 GB RAM, BETA, images too), qwen38-27b: Qwen3.8-27B (unsloth's UD-Q4_K_XL, an 8 GB+ GPU, the FFN partly in RAM under 20 GB), bonsai27b: Ternary Bonsai 2.5 27B (prism-ml's PTQ1_0, an 8 GB+ GPU, text only, BETA), qwen35-9b-qsa: Qwen3.5-9B QSA (zephel01's Q4_K_M, an 11 GB+ GPU, text only), ornith-9b-qsa: Ornith-1.5-9B QSA (the same, text only), mimo-9b: MiMo-V2.6-Distill-Qwen-9B (bartowski's Q4_K_M, an 11 GB+ GPU, text only), mimo-9b-qsa: the same with a QSA indexer (zephel01's), sensenova-loop-9b: SenseNova-Loop-9B (guideboardlabs' Q4_K_M, a 10 GB+ GPU, text only), qwen35: Qwen3.5-122B-A10B (zephel01's IQ3_XXS for a 16 GB GPU + 64 GB RAM), memt-flash: MemT-1.0-Flash (native memory, BETA, NVIDIA only: zephel01's overlay over the v4 32G96G L97 shards, which it sets up too, a 32 GB GPU + 96 GB RAM), memt-nano: MemT-1.0-Nano (native memory, BETA, NVIDIA only: zephel01's Q4_K_M, measured on a 32 GB GPU); "
                    "qwen38: Qwen3.8-Flash-Next, glm: GLM-5.3-Flash (zephel01's GGUFs, --quant picks the PC size)")
    ap.add_argument("--quant", help="v4: a folder of " + REPO + " (default UD-Q2_K_XL, the tested one), or zephel01's 16G64G / 32G64G / 32G96G; v41: 132G "
                    "(default, speed) or 139G (quality); ornith: IQ3_XXS-B, 24G64G, 32G128G (Ornith-1.5-397B) or 16G64G (Ornith-1.5-397B on a 16 GB GPU); ornith35: Q4_K_M (default), Q5_K_M, Q6_K or Q8_0; laguna: UD-IQ4_XS; inkling2mini: UD-Q4_K_XL; inkling2mini-abl: Mix-16G (default) or Mix-12G; ling32air: Q4_K_M; qwen38-27b: any file of unsloth/Qwen3.8-27B-GGUF (UD-IQ1_S .. UD-Q8_K_XL, Q4_0, Q4_1, Q8_0, BF16; default UD-Q4_K_XL); bonsai27b: PTQ1_0 (default, with the MTP block) or Q2_0; qwen35-9b-qsa / ornith-9b-qsa / mimo-9b / mimo-9b-qsa: Q4_K_M (default) or Q8_0; sensenova-loop-9b: Q4_K_M (default), Q3_K_M, Q4_0 or Q8_0; qwen38: 12G32G, 16G64G or 24G64G; glm: 32G96G or 32G128G; memt-flash: 32G96G; memt-nano: Q4_K_M.  Without it, the PC-size file that suits this PC's VRAM + RAM is "
                    "proposed (and taken with --yes; a file set up before is kept); v4 keeps UD-Q2_K_XL and names the one that suits")
    ap.add_argument("--name", help="the model's name in abyss-<name>.json / run-abyss-<name>.sh (default dsv4-<quant>, "
                                   "dsv41-<quant>)")
    ap.add_argument("--data-dir", default=str(ROOT.parent / "Abyss-data"), help="where the models go (default: Abyss-data beside Abyss)")
    ap.add_argument("--tools-dir", default=str(ROOT.parent / "Abyss-tools"), help="the tools folder (default: Abyss-tools beside Abyss)")
    ap.add_argument("--context", type=int, help="the context in tokens (default 65536; memt-flash / memt-nano: 16384)")
    ap.add_argument("--port", type=int, help="the server's port (default 8082; --rewrite-config keeps the config's)")
    ap.add_argument("--backend", choices=("cuda", "hip"), help="cuda (NVIDIA) or hip (AMD: RX 7900 XT / XTX, RX 9070 / "
                    "AI PRO R9700, Ryzen AI Max's Radeon 8060S; ROCm 7). Default: cuda when an NVIDIA GPU is there, else hip")
    ap.add_argument("--gpu", type=int, help="the GPU, as nvidia-smi numbers it (default: the one with the most VRAM)")
    ap.add_argument("--gpus", help="several GPUs for a layer split, as nvidia-smi numbers them (\"0,1\"; the first is "
                                   "the main card), or \"all\": every NVIDIA GPU (default: one GPU, --gpu)")
    ap.add_argument("--no-vision", action="store_true", help="qwen38-27b, ling32air: skip the image encoder (text only)")
    ap.add_argument("--arch", help="the CUDA architecture to compile for (default: the GPU's; 120a on RTX 50)")
    ap.add_argument("--nvcc", help="the CUDA compiler to build with (default: nvcc on PATH, else /usr/local/cuda); a V100 "
                                   "needs a CUDA 12 one, e.g. /usr/local/cuda-12.9/bin/nvcc")
    ap.add_argument("--prebuilt", action="store_true", help="download the prebuilt engine from the GitHub release "
                    "instead of compiling it (taken by itself when nvcc or the C++ compiler is missing; RTX 30 / 40 / "
                    "50, NVIDIA driver 580+, 610+ recommended)")
    ap.add_argument("--download-model", action="store_true", help="download the model without asking")
    ap.add_argument("--yes", action="store_true", help="the defaults for every question (the download still needs --download-model)")
    ap.add_argument("--rewrite-config", action="store_true", help="write abyss-<name>.json again")
    ap.add_argument("--dspark", action="store_true", help="(V4) download DeepSeek's DSpark drafter (6.9 GB) and turn it on in "
                    "the config: drafts checked a step at a time, ~+28%% on swe-bench tasks on an AI PRO R9700 (~6.5 GB "
                    "of VRAM; one GPU, DeepSeek V4 Flash 0731). Asked when the config is new")
    ap.add_argument("--no-dspark", action="store_true", help="turn the DSpark drafter off in the config (the file stays; "
                    "inkling2mini / inkling2mini-abl: its drafter, on when the file is on disk - STRATA_INK_DSPARK=0)")
    ap.add_argument("--dflash", action="store_true", help="(qwen38) download the DFlash 2 drafter (0.65 GB) and turn it on "
                    "in the config instead of the MTP layer: faster code and lists, slower prose (docs/MANUAL.md, "
                    "Qwen3.8-Flash-Next); one GPU.  (qwen38-27b) set up its DFlash 2 drafter (1.14 GB) also where it does "
                    "not fit in VRAM beside the model (it is set up by default where it fits)")
    ap.add_argument("--no-dflash", action="store_true", help="turn the DFlash 2 drafter off in the config (the file stays); "
                    "qwen38-27b: do not set it up")
    ap.add_argument("--tools-only", action="store_true", help="only create / update the Abyss-tools folder")
    a = ap.parse_args()
    if a.dspark and a.no_dspark:
        ap.error("--dspark and --no-dspark together")
    if a.dflash and a.no_dflash:
        ap.error("--dflash and --no-dflash together")
    tools = Path(a.tools_dir).expanduser().resolve()
    if a.tools_only:
        make_tools(tools)
        return 0
    model = MODELS[a.model]
    if a.quant and model["quants"] and a.quant not in model["quants"]:
        ap.error(f"--model {a.model} has the quants {', '.join(model['quants'])}")
    if a.dspark and a.model != "v4":
        ap.error("--dspark: the DSpark drafter is set up for DeepSeek V4 only (V4.1: docs/deepseek41/README.md)")
    if (a.dflash or a.no_dflash) and a.model not in ("qwen38", "qwen38-27b"):
        ap.error("--dflash: the DFlash 2 drafters are Qwen3.8-Flash-Next's and Qwen3.8-27B's (--model qwen38 / qwen38-27b)")
    data = Path(a.data_dir).expanduser().resolve()

    S.step(1, "the PC")
    gpu = S.gpu_info(a.gpu) if a.backend != "hip" else None
    backend = a.backend or ("cuda" if gpu is not None else "hip" if S.amd_gpus() else None)
    if backend is None:
        S.fail("no NVIDIA or supported AMD GPU found", "Abyss runs on NVIDIA GPUs (CUDA), or AMD ones with --backend hip")
    if model.get("nvidia_only") and backend != "cuda":
        # (the memory's kernels build for HIP but have not run there: stopped before ROCm or the engine is touched)
        S.fail(f"{model['title']}'s native memory runs on NVIDIA GPUs only in this version (CUDA; on AMD it is untested)",
               "an NVIDIA GPU (RTX 30 / 40 / 50); on AMD, DeepSeek V4 Flash 0731 (--model v4 --quant 32G96G) or "
               "Ornith-1.5-35B-A3B (--model ornith35) run without the memory")
    hip_extra = None
    if backend == "hip":
        gpus, hip_extra = hip_engine(a)
        S.ok(f"RAM {S.ram_gb():.0f} GB")
    if backend == "cuda" and gpu is None:
        S.fail("no NVIDIA GPU found (nvidia-smi)", "with an AMD GPU: --backend hip")
    gpus = [gpu] if backend == "cuda" else gpus
    if a.gpus and backend == "cuda":
        found = {g["index"]: g for g in S.gpus()}
        want = sorted(found) if a.gpus.strip() == "all" else [int(x) for x in a.gpus.split(",") if x.strip()]
        missing = [i for i in want if i not in found]
        if missing:
            S.fail(f"there is no GPU {missing[0]}: " + ", ".join(f"{i} = {g['name']}" for i, g in found.items()))
        gpus = [found[i] for i in want]
    for g in gpus if backend == "cuda" else []:
        S.ok(f"GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB, compute capability {g['arch']}")
    if model.get("nvidia_only") and len(gpus) > 1:
        S.fail(f"{model['title']} runs on one GPU (a layer split is off for MemT)", "set it up without --gpus (--gpu picks the card)")
    a._prebuilt = backend == "cuda" and use_prebuilt(a, gpus)
    if backend == "cuda" and len(gpus) == 1 and gpu.get("count", 1) > 1:
        S.say(f"  ({gpu['count']} NVIDIA GPUs here: --gpus all (or --gpus 0,1) splits the layers across them)")
    # a model that runs all in VRAM (Qwen3.8-27B): one GPU that holds it, and the context it holds - checked first
    # (no --context: 64K, MemT 16K - the context its numbers were measured at)
    a.context = a.context or model.get("context", 65536)
    ctx = a.context
    # the quant: --quant, else the PC-size file that suits this PC's VRAM + RAM (choose_quant)
    quant = choose_quant(a, a.model, gpus)
    if model.get("vram_ctx"):
        ctx = dense_gpu_check(a, a.model, gpus, backend, pc_memory(gpus, S.ram_gb())[2], quant=quant)
    own = quant in model.get("variants", {})   # (the quant is a PC size and picks zephel01's repo for it)
    if own:
        model = {**model, **model["variants"][quant]}
    if "prefix_tmpl" in model:
        model = {**model, "prefix_file": model["prefix_tmpl"].format(q=quant)}
    repo = model["repo"]
    name = a.name or model["prefix"] + "-" + quant.lower().removeprefix("ud-")
    mdir = data / "models" / repo / quant
    if own and a.model != "ornith" or quant in ("24G64G", "32G128G", "16G64G"):
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is made for a {model['vram']} GB GPU + {model['ram']} GB RAM: with "
             f"VRAM, ~{model['gb']} GB holds every expert; less reads from the SSD)")
        if S.ram_gb() + (gpus[0].get("vram_gb") or 0 if gpus else 0) < model["gb"] * 0.8:
            S.say(f"  (this PC has less: a smaller --quant ({', '.join(model['variants'])}) may suit it better)")
    elif a.model == "v4":
        S.ok(f"RAM {S.ram_gb():.0f} GB (with VRAM, ~100 GB holds every expert of the 97 GB quant; less reads from the SSD)")
    elif a.model == "laguna":
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is ~{model['gb']} GB: a 16 GB GPU + 64 GB RAM holds every expert; "
             "less reads from the SSD)")
    elif a.model in ("inkling2mini", "inkling2mini-abl"):
        gb = {"Mix-12G": 12}.get(quant, model["gb"])   # (the abliterated Mix-12G: 11.6 GB, all in VRAM on a 16 GB GPU)
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is ~{gb} GB: all in VRAM on a {16 if gb < 17 else 24}-32 GB GPU; "
             "a 12-16 GB GPU + 32 GB RAM holds the rest in RAM)")
    elif a.model == "ling32air":
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is ~{model['gb']} GB: a 24-32 GB GPU + 64-96 GB RAM holds every "
             "expert; less reads from the SSD)")
    elif a.model == "ornith35":
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is ~{model['gb']} GB: VRAM + RAM hold it all on an 8-16 GB GPU + "
             "32 GB RAM PC)")
    elif a.model == "memt-flash":
        S.ok(f"RAM {S.ram_gb():.0f} GB (made for a 32 GB GPU + 96 GB RAM, as its base L97: with VRAM, ~100 GB holds every "
             "expert; less reads from the SSD)")
        if (gpus[0].get("vram_gb") or 0) < 30:
            S.warn(f"MemT-1.0-Flash is measured on a 32 GB GPU only (RTX 5090): writing a memory needs ~4.1 GB at the end of "
                   "the expert pool (a 32 GB GPU has 13 GB there) - with less the engine refuses the write with a message")
    elif a.model == "memt-nano":
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is ~{model['gb']} GB: measured all in VRAM on an RTX 5090, 32 GB)")
        if (gpus[0].get("vram_gb") or 0) < 30:
            S.warn(f"MemT-1.0-Nano is measured on a 32 GB GPU only (RTX 5090): on this smaller one the experts that do not "
                   "fit go to RAM as for Ornith-1.5-35B-A3B (the same shapes); its speed and the memory there are not measured")
    elif model.get("vram_ctx"):
        gb = (model.get("quant_gb") or {}).get(quant, (model["gb"],))[0]
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file, ~{gb:.0f} GB, runs in VRAM: RAM holds the server and the "
             "page cache)")
    elif a.model in ("ornith", "qwen35"):
        S.ok(f"RAM {S.ram_gb():.0f} GB (made for a 16 GB GPU + 64 GB RAM: with VRAM, ~60 GB holds every expert of the "
             f"{model['gb']} GB file; less reads from the SSD)")
    else:
        S.ok(f"RAM {S.ram_gb():.0f} GB (made for a 32 GB GPU + 96 GB RAM: the experts that do not fit there and the "
             "Engram table are read from the SSD - keep the model on a fast NVMe)")

    S.step(2, "Python packages and llama.cpp's source")
    S.pip_install(S.PY_PACKAGES, ", ".join(S.PY_PACKAGES))
    why = S.llama_cpp_reasons(gpu_engine=not a._prebuilt)   # (images fetch it in their own step)
    S.ok(f"llama.cpp: {S.get_llama_cpp(why)} ({'; '.join(why)})" if why else "llama.cpp's source: not needed")

    S.step(3, "the engine")
    if backend == "cuda" and a._prebuilt:
        prebuilt_engine(gpus)
    elif backend == "cuda":
        build_engine(a.arch, gpus, a.nvcc)

    S.step(4, f"the model: {repo} {quant}")
    files = quant_files(quant, repo, (model.get("quant_folders") or {}).get(quant, model.get("folder")),
                        model.get("prefix_file", "") if quant not in (model.get("quant_folders") or {})
                        else model["prefix_file"].rstrip(".") + "-")
    groups = [(repo, mdir, files)]
    if model.get("base"):
        # (an overlay - MemT-1.0-Flash: its base's shards first, in the base's own setup folder, so a base set up before
        # is only checked; then the overlay's shard 1, and the base's shards linked beside it as shards 2..N)
        bkey, bq = model["base"]
        bm = {**MODELS[bkey], **MODELS[bkey]["variants"][bq]}
        bdir = data / "models" / bm["repo"] / bq
        bfiles = quant_files(bq, bm["repo"], bm["folder"], bm["prefix_file"])
        S.say(f"  {model['title']} is one file over {bm['title']} {bq} ({bm['folder']}): both are needed")
        groups.insert(0, (bm["repo"], bdir, bfiles))
    fetch_files(a, data, groups)
    if model.get("base"):
        overlay_links(mdir, Path(files[0][0]).name, bdir, bfiles)
    mtp = None
    if (model.get("quant_gb") or {}).get(quant, (0, True))[1] is False:
        mtp = side_file(a, repo, DENSE27_MTP, mdir, "the NextN draft block (this file has none)")
    if a.model in ("mimo-9b", "mimo-9b-qsa"):
        mtp = borrowed_mtp(a, MIMO9_MTP_REPO, MIMO9_MTP, mdir, model["title"])

    S.step(5, "the pack")
    pack = mdir / ("pack-dsv4" if a.model == "v4" else "pack")
    pack_args = model.get("pack_args", ["--compat-bf16"])
    if (pack / "native_experts.txt").exists() and (pack / "tokenizer" / "chat_template.jinja").exists():
        S.ok(f"pack: {pack} (already made)")
    else:
        first = mdir / Path(files[0][0]).name
        S.run([sys.executable, ROOT / "tools" / "iq_pack.py", "--gguf", first, "--out", pack, *pack_args], quiet=True)
        S.ok(f"pack: {pack}")

    S.step(6, "the run config")
    cfg_path = ROOT / f"abyss-{name}.json"
    fresh = a.rewrite_config or not cfg_path.exists()
    had = None
    if cfg_path.exists():
        try:
            had = (json.loads(cfg_path.read_text(encoding="utf-8")).get("env") or {}).get("STRATA_DS4_DSPARK")
        except (OSError, ValueError):
            pass
    extra = dict(hip_extra or {})
    if model["env"]:
        extra["env"] = {**model["env"], **extra.get("env", {})}
    if a.model in ("qwen38", "glm", "ornith35", "laguna", "ling32air") and S.ram_gb() < 160:
        # the RAM tier the model cards were measured with: 64 GB PC -> 46, 32 GB -> 20, 96 GB -> 60 (the rest for the OS,
        # the page cache - Qwen3.8's PLE table is read from the SSD through it - and the server)
        ram = S.ram_gb()
        extra.setdefault("env", {})["STRATA_GLM_RAM_GB"] = str(max(12, round(ram) - (12 if ram < 48 else 18 if ram < 80 else 34)))
    if a.model == "ornith" and quant == "16G64G" and S.ram_gb() < 160:
        # Ornith-1.5-397B 16G64G: the model card's RAM tier (44 GB on a 64 GB PC: the rest for the OS and the page cache
        # the SSD experts go through)
        extra.setdefault("env", {})["STRATA_GLM_RAM_GB"] = str(max(16, round(S.ram_gb()) - 20))
    if a.model == "v41" and S.ram_gb() < 160:
        # a RAM tier that leaves ~34 GB to the page cache and the rest of the PC: V4.1's Engram rows and SSD experts keep
        # the page cache full, and without a cap systemd-oomd stops the server or the session that started it (the
        # card's numbers: 60 GB on 94 GB)
        extra.setdefault("env", {})["STRATA_GLM_RAM_GB"] = str(max(16, round(S.ram_gb()) - 34))
    if mtp is not None:
        # (the engine also finds MTP/ beside the GGUF by itself; the config names it)
        extra.setdefault("env", {})["STRATA_GLM_MTP_GGUF"] = str(mtp)
    cfg, _ = write_config(name, pack, ctx, a.port, a.rewrite_config,
                          gpus if len(gpus) > 1 and backend == "cuda" else None, extra or None, model)

    # the engine's per-GPU kernel choices measured on these GPUs (./abyss.sh --gpu-tune; under a minute a GPU, no
    # model): a faster one goes into the config's env.  Never stops the setup
    S.say()
    S.say("  Checking the engine's per-GPU kernel choices on this GPU (./abyss.sh --gpu-tune) ...")
    try:
        import abyss as M   # noqa: E402  (constants and functions only)
        M.gpu_tune(argparse.Namespace(backend=backend, gpu=None, gpus=None))
    except (SystemExit, Exception) as e:   # (fail() exits)
        S.warn(f"the GPU check did not run ({e or 'see above'}): the engine's own choices stay")

    S.step(7, "the DSpark drafter")
    if a.model == "v4" and own and not a.dspark:
        # (the model cards: on a 32 GB GPU the drafter's ~6.5 GB push experts to the SSD - 69.1 -> 62.5 tok/s at 4K)
        S.ok("DSpark drafter: off - zephel01's V4 GGUFs run faster without it on a discrete GPU (--dspark turns it on)")
    elif a.model == "v4":
        dspark_step(a, data, cfg, fresh, had, gpus, backend)
    elif a.model in ("inkling2mini", "inkling2mini-abl"):
        ink_dspark_step(a, mdir, cfg, gpus, backend)
    else:
        S.ok("DSpark drafter: set up for DeepSeek V4 only" + (" (V4.1: by hand, docs/deepseek41/README.md)" if a.model == "v41"
             else " (Ornith drafts with its own NextN block from the shards)" if a.model in ("ornith", "ornith35")
             else " (Qwen3.8 drafts with its own MTP block from the shards)" if a.model == "qwen38"
             else (" (Qwen3.8-27B drafts with unsloth's separate NextN block, MTP/)" if mtp else
                   " (Qwen3.8-27B drafts with its own NextN block from the file)") if a.model == "qwen38-27b"
             else (" (Ternary Bonsai drafts with the MTP block in its PTQ1_0 file)" if quant == "PTQ1_0" else
                   " (Ternary Bonsai Q2_0: no draft block in the file)") if a.model == "bonsai27b"
             else " (Qwen3.5-9B QSA drafts with its own NextN block from the file)" if a.model == "qwen35-9b-qsa"
             else " (Ornith-1.5-9B QSA drafts with its own NextN block from the file)" if a.model == "ornith-9b-qsa"
             else (" (MiMo-V2.6-Distill-Qwen-9B drafts with Qwen3.5-9B's NextN block, MTP/)" if mtp else
                   " (MiMo-V2.6-Distill-Qwen-9B: no draft block - n-gram drafts)") if a.model in ("mimo-9b", "mimo-9b-qsa")
             else " (SenseNova-Loop-9B: no draft block - n-gram drafts)" if a.model == "sensenova-loop-9b"
             else " (Ling-3.2-air: its MTP block is not used yet)" if a.model == "ling32air"
             else " (MemT: drafting is off - the draft checks do not read the memory)" if model.get("nvidia_only")
             else ""))
    if a.model == "qwen38":
        dflash_step(a, data, cfg, gpus)
    if a.model == "qwen38-27b":
        need = dense_need(a.model, ctx, quant)
        resident = bool(gpus) and need is not None and (gpus[0].get("vram_gb") or 0.0) - need >= 1.5
        vision_27b_step(a, repo, mdir, mdir / Path(files[0][0]).name, cfg, gpus, backend, resident)
        # (the drafter: on where the whole file, its context and the drafter fit - an NVIDIA GPU; dflash2.cu is CUDA)
        fits = (resident and backend == "cuda" and
                (gpus[0].get("vram_gb") or 0.0) - need >= 1.5 + DFLASH27_GB)
        dflash27_step(a, data, cfg, gpus, fits)
    if a.model == "ling32air":
        vision_ling_step(a, repo, mdir, mdir / files[0][0], cfg, backend)

    S.step(8, "Abyss-tools")
    make_tools(tools)

    S.say()
    S.say(f"Done. Start it with:  {tools / 'abyss.sh'} start {name}")
    S.say(f"then open http://127.0.0.1:{json.loads(cfg.read_text(encoding='utf-8')).get('port', 8082)}/  (the manual: {ROOT / 'docs' / 'MANUAL.md'})")
    if a.model.startswith("memt-"):
        sec = "memt-10-flash-native-memory-beta" if a.model == "memt-flash" else "memt-10-nano-native-memory-beta"
        S.say('The memory: "memory": {"id": "<name>", "commit": true} in a chat request reads that memory and writes the '
              f"turn into it; without it the model answers without one (docs/MANUAL.md#{sec})")
    if not model.get("vram_ctx"):   # (a model all in VRAM has no CPU lane or PCIe split to tune)
        S.say(f"Once, with it stopped: {tools / 'abyss.sh'} --calibrate  tunes the CPU lane / PCIe split for this PC "
              "(~10-15 minutes; +12% on an RTX 4090, issue #39)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
