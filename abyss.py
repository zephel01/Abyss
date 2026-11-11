#!/usr/bin/env python3
"""Abyss - set up and run big MoE models (GLM-5.3-Flash, DeepSeek V4 / V4.1, Qwen3.8-Flash-Next, Ornith) on your own
GPU(s) + RAM + SSD.  CUDA: Linux; Windows (experimental).  HIP: gfx1100/gfx1201/gfx1151, Linux and Windows.
DeepSeek V4 / V4.1, Qwen3.8 and Ornith are installed by ./setup-deepseek.sh (docs/MODELS.md); this file runs them all.

Setup and start:
    ./abyss.sh                 the first run sets everything up and starts the dashboard; later runs just start it
    ./abyss.sh --setup         set up again (other GPUs, another context length, another model folder)
    ./abyss.sh --check         nothing installed yet: only check this PC (installed: the model check below)
    ./abyss.sh --config abyss-<name>.json      start another installed model (default: the last one used)
    ./abyss.sh --backend hip --gpu 0 --check   check an AMD card with system ROCm 10

Abyss Terminal (watch the server: speed, where a token's time goes, GPUs, SSDs, memory, requests; start / stop / switch):
    ./abyss.sh --terminal                      this PC's server (l: the Speed / Devices layouts, ?: keys, q: quit)
    ./abyss.sh --terminal --layout devices     open on the Devices layout (every GPU / SSD / CPU)
    ./abyss.sh --terminal --url http://192.168.1.20:8082 --key-env ABYSS_KEY   a server on another PC
    ./abyss.sh --terminal --demo r9700         recorded readings, no server (also: lowspec split multi4 multi8 multi10
                                               loading starting down)
    ./abyss.sh --terminal --once / --json      one reading as text / JSON, then exit

Coding tools (Claude Code, OpenCode) against the server that runs now:
    ./abyss.sh --connect claude                print the settings (address, key, model name, context come from the server)
    ./abyss.sh --connect opencode --write      write them (a launcher script / opencode.json), asked first, a backup kept
    ./abyss.sh --connect claude --check        one prompt and one tool call the way the tool sends them

Speed and tuning:
    ./abyss.sh --bench                         a standard speed test (Abyss stopped): decode, prefill, the bottleneck,
                                               a quality check; --deep finds the bottleneck, --full adds a 32k prompt
    ./abyss.sh --calibrate                     tune the CPU lane for this PC (~10-15 minutes)
    ./abyss.sh --gpu-tune                      measure the per-GPU kernel choices (about a minute a GPU)
    ./abyss.sh --check                         a stability check of the installed model (Abyss stopped, ~15-30 min):
                                               long prompts up to the context, a 10+ turn conversation, Japanese,
                                               code run against tests, a tool call, the drafter on / off, speeds,
                                               what limits this PC; --quick ~5 min, --long adds the biggest prompts
    ./abyss.sh --check --finance               Japanese finance rules on synthetic accounts (17 items, ~10-30 min;
                                               market-independent only - not investment advice), abyss-finance.txt
    ./abyss.sh --report                        write abyss-report.txt to attach to an issue (nothing is sent)
    ./abyss.sh --repack / --rebuild            build the model's pack / the engine again

On Windows START-ABYSS.bat takes the same options.  Both make the private Python environment (.venv) and run this
file.  Its helpers (setup.py) do the PC checks, pip, llama.cpp's source and resumable downloads.

In a terminal the setup runs on a screen of its own (tools/setup_tui.py): the steps, what runs now with a progress
bar, and the questions as menus - then Maya runs on the same screen, and every later start too: its dashboard (the
Monitor's numbers from the engine's GET /metrics), its server's output, Ctrl+C stops it.  Its library, Textual, comes
with Maya (third_party/wheels) and goes into .venv from there - nothing is downloaded.  --plain or a pipe: plain text,
and Maya in the terminal as before; --yes: the setup in plain text.

What the first run does (each step is skipped when it is already done):

  1. checks the PC: NVIDIA GPU(s) of compute capability 7.0+, driver, CUDA toolkit (nvcc), the C++ compiler (g++;
     on Windows Visual Studio 2022's Build Tools), CMake, RAM, CPU; HIP checks AMD gfx1100/gfx1201/gfx1151 and ROCm 10 instead
     (on Windows AMD's ROCm SDK wheels in .venv, offered when they are not there)
  2. asks: which GPUs (one, or several that split the layers), how much context, which model to download (Maya-S,
     Maya-S24, Maya-M or Maya-L; or turboderp's EXL3 quant, experimental)
  3. Python packages into .venv, llama.cpp's source at the pinned commit (it lists them and asks first)
  4. compiles the engine (`build/strata`, or `build-hip/strata`) for your GPU(s): 10-30 minutes, once
  5. the model: GGUF files you already have (--gguf-dir), or a download it shows you first - the exact commands
     and the size - and starts only after you answer y (or pass --download-model)
  6. builds the pack (the engine's index of the model files) inside the model folder
  7. images: compiles the vision encoder (`build-vision/bin/strata-vision`) and fetches the model's vision files
     (1.1 GB, shown and asked first like the model; --no-vision and HIP skip it)
  8. writes maya-<model>.json and run-maya-<model>.sh (.bat on Windows) - with the settings tuned for this PC
     earlier, or it offers the tuning (./abyss.sh --calibrate does it any time) - and starts the dashboard on
     http://127.0.0.1:8080

The tuning (tools/calibrate_glm.py): decode speed measured with a few splits of the RAM-tier
experts between the CPU and the PCIe link, and with fewer CPU threads, in one engine run (~10-15 minutes, the model
loads first); a setting is kept when it is more than 3% faster than the engine's own choice.  The result goes into the
config's "env" (STRATA_GLM_PCIE_SHARE, STRATA_GLM_CPU_LANE) and into ~/.config/project-maya/calibration.json for this
PC, model and context, so a setup again keeps it.

Nothing is installed system-wide: a missing tool is reported with the command that installs it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
import setup as S  # noqa: E402  Strata's installer: PC checks, pip, llama.cpp, downloads (nothing runs on import)
from setup import ask, fail, ok, run, say, step, warn  # noqa: E402
import calibrate_glm as CAL  # noqa: E402  the CPU lane's tuning for this PC (./abyss.sh --calibrate)
# where a token's time goes, the bottleneck and its advice: shared with Abyss Terminal (tools/bench_rules.py)
from bench_rules import (BENCH_ADVICE, BENCH_ADVICE_HIP, GAP_LINE, SPEC_LINE, STAT_LINE, _mean, advice,  # noqa: E402,F401
                         bench_records, bench_verdict, lane_name as log_lane_name)

WIN = S.WIN
ME = "START-ABYSS.bat" if WIN else "./abyss.sh"      # how this is started, for the messages
UPDATE_EXIT = 75                                   # the server's exit after the dashboard updated Maya (serve/update.py)
ROOT = S.ROOT
BUILD = ROOT / "build"
EXE = BUILD / ("strata.exe" if WIN else "strata")
STAMP = BUILD / "MAYA-BUILD.json"                  # what the engine in build/ was compiled from and for
VBUILD = ROOT / "build-vision"
VEXE = VBUILD / "bin" / ("strata-vision.exe" if WIN else "strata-vision")  # the image encoder (llama.cpp's mtmd)
VSTAMP = VBUILD / "MAYA-BUILD.json"
MIN_CC = 70                                        # Volta (V100) and newer (the GLM path; see arch_setting)
MAX_GPUS = 16                                      # the layer split's parts at most (glm_model.hpp, kMaxParts)
# every HIP build targets the validated Radeons: RX 7900 (gfx1100), RX 9070 / AI PRO R9700 (gfx1201), Strix Halo (gfx1151)
HIP_BUILD_ARCHS = ("gfx1100", "gfx1201", "gfx1151")
# ... and setup accepts their siblings, which run the same kernels (cmake/hip_backend.cmake: community / unvalidated),
# adding the one it finds to the build: RX 7800 XT / 7700 XT (gfx1101), RX 7600 (gfx1102), Radeon 890M (gfx1150),
# RX 9060 (gfx1200)
HIP_ARCHS = HIP_BUILD_ARCHS + ("gfx1101", "gfx1102", "gfx1150", "gfx1200")
PY_PACKAGES = list(S.PY_PACKAGES)                  # (pillow: pictures in formats other than JPEG/PNG/BMP/GIF)
CONTEXTS = [8192, 32768, 65536, 131072]
DEFAULT_CONTEXT = 32768
MODEL_NAME = "glm-5.3-flash"
SAMPLING = {"temperature": 1.0, "top_p": 0.95}     # the dashboard's and the API's defaults for requests that set none
EFFORT = "high"                                    # thinking level for requests that name none (GLM's High)
HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
# The models the installer can download: Project Maya's own quants, made from Z.ai's FP8 release (their model card
# has the measurements against it).
# Another glm5-next GGUF can be used with --gguf-dir (experimental).  "folder": the files' folder in the repo ("" =
# its top).  "file": the names on Hugging Face, with the quant label its file list groups (and adds up) them by -
# one label per model; "was": the names they had there before (v1.0.18 and earlier: no label; v1.0.19: Maya-S24 as
# IQ2_XXS, which Hugging Face added to Maya-S's) - a download under them is used as it is.  "sha256" per file name (the
# Hugging Face one): every download is verified; a list holds every version published (v1.0.28: the first shards' header
# names the architecture "glm5-next", as llama.cpp does - the data is the same, so a download of either is good).  "vision": the image encoder's files (the mmproj,
# made from the official vision tower, and the tokenizer it reads its markers with) - from "repo" / "revision" when
# it names them, else the model's own repo.
MODELS = {
    "Maya-S-v2-IQ2_XXS": {
        "about": "Maya-S, Project Maya's compact quant: error-feedback-rounded IQ2_XXS gate/up experts, "
                 "IQ2_S/IQ3_XXS down projections, Q6_K attention, the MTP draft block, made from Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-S-v2-IQ2_XXS",
        "file": "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-{i:05d}-of-{n:05d}.gguf", "shards": 3, "download_gb": 96.5,
        "sha256": {
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf":
                ["e56ef50efd71d81ddf6249e08bd2ef5af96ed348c2b7a004491d46168d0ecaa7",    # "glm5-next" (v1.0.28)
                 "a507f2b7b25e04624ee55c631d3b281caea7cfffc747e5247970cd5a7ea91b3f"],   # "glm5next", before
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00002-of-00003.gguf":
                "2d65d88a69f8dc124c8d24bd33b33161ddab218918b4253ad4f500aeede84df5",
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00003-of-00003.gguf":
                "a6a981c4fee7a53d78bdbd97d8f465d48ddf439cf938ff90617f395e0351b5a6"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-S24": {
        "about": "Maya-S24, Maya-S with Q4_K attention and shared experts: about 1.5 GB less that stays on the GPU, so "
                 "24 GB cards hold more experts (decode about 14% faster there, 11% on 32 GB), close to Maya-S in "
                 "quality",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-S24",
        "file": "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-S24-IQ2_XXS-{i:05d}-of-{n:05d}.gguf",
                "GLM-5.3-Flash-Maya-S24-{i:05d}-of-{n:05d}.gguf"], "shards": 3, "download_gb": 94.7,
        "sha256": {
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00001-of-00003.gguf":
                ["8ebd5c74bcf65ece2a8699b7b6fd2b8d177f350ed252cee252fdd99b86a0138c",
                 "3dc347757686c1435eae36c4872f5c151191cc5063bc188ce699b97700aae076"],
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00002-of-00003.gguf":
                "4bd445da3a0128a9c5b32228c924e0a622aa4a132c7a8dc9a2c207a4beea8e34",
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00003-of-00003.gguf":
                "a6fd4e88007ac49b431c7d02be34e43ab7a09224d53a0c52782327613cb41917"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-M": {
        "about": "Maya-M, Project Maya's larger quant: error-feedback-rounded IQ2_S gate/up experts, IQ3_XXS down "
                 "projections (IQ3_S in the most sensitive layers), Q6_K attention, the MTP draft block, made from "
                 "Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-M",
        "file": "GLM-5.3-Flash-Maya-M-IQ2_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-M-{i:05d}-of-{n:05d}.gguf"], "shards": 3, "download_gb": 116.0,
        "sha256": {
            "GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf":
                ["3f740d1b1022b4a0ddad50cd861bb071dd415bde94afa1292d0476641186c638",
                 "3ac0f066ec45af3432d59b33de49bdfb29156432b627240b35769c7d02cc6c02"],
            "GLM-5.3-Flash-Maya-M-IQ2_S-00002-of-00003.gguf":
                "285951d2afa0cd98285b03d0dc4aa68d83daf1a6f4a2594fe40b0cdd27ade485",
            "GLM-5.3-Flash-Maya-M-IQ2_S-00003-of-00003.gguf":
                "ebf1ce713f71207747e10eeebed87597969d8b5e1d9dd817420d2c2f7ca51e0e"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-L": {
        "about": "Maya-L, Project Maya's largest quant: error-feedback-rounded IQ3_S gate/up experts, IQ4_XS down "
                 "projections (Q5_K in the most sensitive layers), Q6_K attention, the MTP draft block, made from "
                 "Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-L",
        "file": "GLM-5.3-Flash-Maya-L-IQ3_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-L-{i:05d}-of-{n:05d}.gguf"], "shards": 4, "download_gb": 156.3,
        "sha256": {
            "GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf":
                ["c41aeaf3a3e0022150b7c355e7f4df6c92ba68c4794e3e827e02e56b54886b9e",
                 "5fc82a6c9af4c6898e8d45cf32964f9a7b10640caf51e82be735d2d50e9f3d45"],
            "GLM-5.3-Flash-Maya-L-IQ3_S-00002-of-00004.gguf":
                "351d59366afb7be7dfc3fb7af91281f7ccf00760b17dc45f82090325a7b34f64",
            "GLM-5.3-Flash-Maya-L-IQ3_S-00003-of-00004.gguf":
                "3685962fdfeaf130b74f98d3ce703ee33d818aa9991725932cb305b7c48db8af",
            "GLM-5.3-Flash-Maya-L-IQ3_S-00004-of-00004.gguf":
                "1af62cd72de6460c85369d988abbb790d6ec3a97507df51faf9bf32aa2b600d0"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    # turboderp's EXL3 (exllamav3) quant, read in place from its safetensors: the pack (tools/exl3_pack.py) puts a
    # small GGUF of the unquantized tensors beside them.  "files": every file of the branch with its size (what is
    # downloaded, and how a download is known whole); "revision": the branch's commit the hashes belong to.
    "EXL3-sc_2.10bpw": {
        "about": "turboderp's self-calibrated EXL3 quant at 2.10 bits per weight (exllamav3's trellis format, "
                 "which the engine reads in place): GGUF-sized, better text prediction than Maya-S-v2 on our "
                 "test text, slower on an APU's iGPU lane; no MTP draft block - experimental, NVIDIA GPUs only",
        "format": "exl3", "repo": "turboderp/GLM-5.3-Flash-exl3", "branch": "sc_2.10bpw",
        "revision": "8d2eca0ed5b695108249308bcdc2af0636d5045f", "folder": "", "download_gb": 88.3,
        "files": {
            "LICENSE": 1070, "README.md": 7238, "chat_template.jinja": 8617, "config.json": 86529,
            "generation_config.json": 194, "model-00001-of-00012.safetensors": 8567975136,
            "model-00002-of-00012.safetensors": 7765742994, "model-00003-of-00012.safetensors": 7765756986,
            "model-00004-of-00012.safetensors": 7765756986, "model-00005-of-00012.safetensors": 7765756986,
            "model-00006-of-00012.safetensors": 7765756986, "model-00007-of-00012.safetensors": 7765756986,
            "model-00008-of-00012.safetensors": 7765756986, "model-00009-of-00012.safetensors": 7765756986,
            "model-00010-of-00012.safetensors": 7765756986, "model-00011-of-00012.safetensors": 6303367524,
            "model-00012-of-00012.safetensors": 3427783643, "model.safetensors.index.json": 16343097,
            "processor_config.json": 909, "quantization_config.json": 47905716, "tokenizer.json": 20217442,
            "tokenizer_config.json": 761},
        "sha256": {
            "LICENSE":
                "30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1",
            "README.md":
                "4f46db37e6a7cbe3a494d6d7f616a96f4fffe6d1de68be361199a715c8d928fc",
            "chat_template.jinja":
                "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3",
            "config.json":
                "31146dd4192fb1dce263cb14d27f8684fe3ed17f7acd355e8d1ae237c7b024cb",
            "generation_config.json":
                "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f",
            "model-00001-of-00012.safetensors":
                "4cb1891ae8eb9ab0269f049d0ba5e324141b488c8413155943435c77fd1bed72",
            "model-00002-of-00012.safetensors":
                "107a5539bf0c324abb4b5417ef6c6207c5b3df8b84c5a78326caefb9df5ca4a1",
            "model-00003-of-00012.safetensors":
                "a38b16e3754ea271e300bff75a3123e695b25412760127b6d8e47ec0791bd3da",
            "model-00004-of-00012.safetensors":
                "1ef9a817cccbbc803fa0113ffd9ca1a9aed86ac990d95cc1550335175e9bc569",
            "model-00005-of-00012.safetensors":
                "31547000b3892d3469a61aa234b974dcfd4067f70c18b815d00efcf7399b8de5",
            "model-00006-of-00012.safetensors":
                "cf6b42119f6ef7c8af98b00a4c778ae879d04b938872a6e8f401bedfd95018d7",
            "model-00007-of-00012.safetensors":
                "6a6d8a2314409a7ee5281740ab31a85ca9fee0882e159f2d207a74186b88d452",
            "model-00008-of-00012.safetensors":
                "55318a0b50e8dcff5da17336055008494f21790fc58125a53cbc19df98376526",
            "model-00009-of-00012.safetensors":
                "6576557ce2bcba599eed092d420916d299ca09cc6bf0b860c7cd49a8c857fe33",
            "model-00010-of-00012.safetensors":
                "08f2437835c29a31b1f9203d0bd4b6ab47b67711656bf4ad4005683d6e868b00",
            "model-00011-of-00012.safetensors":
                "3135e401817587e188b9137a364cf73534e45224dd2a061285b3377c52f86330",
            "model-00012-of-00012.safetensors":
                "a2b5957a6b398778f1edc09df925e8c57416756cb2b0569bb82abbe13d1b753d",
            "model.safetensors.index.json":
                "d0721e726249ab0f9e10a3c5e3329bb2f1af4de31cd20865386ece9866f4c337",
            "processor_config.json":
                "aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed",
            "quantization_config.json":
                "c92193014ed160c5aff709b7fc53bb8378982e4e6f9e4fb2397d6b7a6b87c1af",
            "tokenizer.json":
                "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
            "tokenizer_config.json":
                "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"},
        "vision": {
            "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "vision",
            "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf", "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "EXL3-3.05bpw": {
        "about": "turboderp's EXL3 quant at 3.05 bits per weight (exllamav3's trellis format, which the engine reads "
                 "in place from its safetensors), with the MTP draft block - experimental, NVIDIA GPUs only",
        "format": "exl3", "repo": "turboderp/GLM-5.3-Flash-exl3", "branch": "3.05bpw",
        "revision": "332ab457b709b7ba30dd9a448be5de03b80a7ac9", "folder": "", "download_gb": 125.3,
        "files": {
            "config.json": 86509, "LICENSE": 1070, "README.md": 7238, "chat_template.jinja": 8617,
            "generation_config.json": 194, "kpool_aux.safetensors": 11548768,
            "model-00001-of-00015.safetensors": 8589636152, "model-00002-of-00015.safetensors": 8588356452,
            "model-00003-of-00015.safetensors": 8590179904, "model-00004-of-00015.safetensors": 8588365508,
            "model-00005-of-00015.safetensors": 8588365508, "model-00006-of-00015.safetensors": 8588365500,
            "model-00007-of-00015.safetensors": 8590186912, "model-00008-of-00015.safetensors": 8588365508,
            "model-00009-of-00015.safetensors": 8588365508, "model-00010-of-00015.safetensors": 8588365500,
            "model-00011-of-00015.safetensors": 8590186912, "model-00012-of-00015.safetensors": 8588365508,
            "model-00013-of-00015.safetensors": 8588365508, "model-00014-of-00015.safetensors": 8590694168,
            "model-00015-of-00015.safetensors": 2059162376, "model.safetensors.index.json": 15960981,
            "mtp.safetensors": 2862645889, "processor_config.json": 909, "quantization_config.json": 47905678,
            "tokenizer.json": 20217442, "tokenizer_config.json": 761},
        "sha256": {
            "config.json":
                "0bef79e79b28fa9c51d57c49bd53adc59005c916832879ad3f0862e284a7bc06",
            "LICENSE":
                "30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1",
            "README.md":
                "4f46db37e6a7cbe3a494d6d7f616a96f4fffe6d1de68be361199a715c8d928fc",
            "chat_template.jinja":
                "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3",
            "generation_config.json":
                "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f",
            "kpool_aux.safetensors":
                "39053cd051c75e81b7e608818111a4b4c3c57da540599089419fe11b51ca02d3",
            "model-00001-of-00015.safetensors":
                "6b6b96190a5114a55d058e3f4ae8a4350ac334dfcf804f61c39d5c6e40340ba4",
            "model-00002-of-00015.safetensors":
                "29fda4953918a156758169afa120929fdd35407860eddf5cad0cb747f65475ed",
            "model-00003-of-00015.safetensors":
                "9844130df84df3138706a0a708775c8d010a7ed370539880e327cac6bec7f2b7",
            "model-00004-of-00015.safetensors":
                "4f2a6fb68fb6cceab231d9c4a54b96174b4ec2b1bbf5cf042a6e4be37c739e29",
            "model-00005-of-00015.safetensors":
                "73cd6d0b5ba7f6d0ce23982729c6342d38114dd89d8a884a2fa08133ef52b47a",
            "model-00006-of-00015.safetensors":
                "160fcfc258b61c7aa7cbc27c4d6a33daee8bfe6dbb0e7dd5e73ce901789f391e",
            "model-00007-of-00015.safetensors":
                "a94f480c4256b7e5311b5384574471caff4f4fc0def8c3e1c57c411222cb1ce4",
            "model-00008-of-00015.safetensors":
                "e513ba853f8b8cb7dac8f17ff9f281787a63cc67daea15fc288591f93eec62f5",
            "model-00009-of-00015.safetensors":
                "825928573f534c09dc8e157356359bc525e883933777fa0d09d7a30a3f13d8b0",
            "model-00010-of-00015.safetensors":
                "f82170395c010c1b40b3695bcca439ac54445b268936ee71eb48c28f347d0d0d",
            "model-00011-of-00015.safetensors":
                "240ece46708e5c41aaf365bb2659ed7da2134bc12d9fa9b5393555c103228c6f",
            "model-00012-of-00015.safetensors":
                "2738042b40d05e144e54ed3c1013831d7dc156b694a1d68a49d55b25369ac8d9",
            "model-00013-of-00015.safetensors":
                "cf097956efdf8c307a6e560786445f1715c7bba9b2bddfb300f7346b1e1eb3ba",
            "model-00014-of-00015.safetensors":
                "1add7e6d8f4866e5aeecf77ac9ca2aacce3f071455fa9c167fa9bc678551c923",
            "model-00015-of-00015.safetensors":
                "d14a0087710b04a680f2d2a8769fa26f4e53dd3c1d5ce5391798ba25da1947f4",
            "model.safetensors.index.json":
                "559538d5006f757c809ef0a545062c2f79abb0140667399d590af0da96f26735",
            "mtp.safetensors":
                "7274f06d75842edfa258830c836fd790a19324513134e05cd1e931929c4c384d",
            "processor_config.json":
                "aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed",
            "quantization_config.json":
                "308d9a4b28407e2ada887f512bc8793c608586a73d9e0108ca258220ac714280",
            "tokenizer.json":
                "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
            "tokenizer_config.json":
                "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"},
        # the pictures: the vision tower and its tokenizer are the same in every GLM-5.3-Flash quant
        "vision": {
            "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "vision",
            "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf", "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
}
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
GLM_ARCHS = ("glm5-next", "glm5next")             # llama.cpp's spelling and unsloth's
PACK_FILES = ("index.txt", "native_experts.txt", "dense.bin", "tokenizer/vocab.json", "tokenizer/merges.txt",
              "tokenizer/token_type.json", "tokenizer/chat_template.jinja")


# ------------------------------------------------------------------------------------------------ small helpers
def hf_url(repo: str, revision: str, folder: str, file: str) -> str:
    """A file's download URL on Hugging Face (folder "" = the repo's top)."""
    return HF.format(repo=repo, revision=revision, path=f"{folder}/{file}" if folder else file)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def configs() -> list:
    """The installed models' configs, the most recently used first - not the dashboard's settings the server keeps
    beside each (<config>.shared-settings.json), which a restart took for a config of the CUDA backend."""
    return sorted((p for p in ROOT.glob("maya-*.json") if not p.name.endswith(".shared-settings.json")),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def mem_gb() -> tuple:
    """(total, available) RAM in GiB - the engine sizes its RAM tier from MemAvailable (on Windows ullAvailPhys)."""
    if WIN:
        m = S._memory_status()
        return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024 / 2**30
    except (OSError, ValueError):
        pass
    return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)


def existing(path: Path) -> Path:
    """`path`, or its nearest folder that exists (where it would be created)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def rotational(path: Path) -> bool:
    """True when `path` is on a spinning disk (best effort: findmnt + lsblk; on Windows the drive's MediaType)."""
    if WIN:
        drive = existing(path).drive[:1]
        return drive.isalpha() and S.out(
            ["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Where-Object DeviceId -eq "
             f"(Get-Partition -DriveLetter {drive}).DiskNumber | Select-Object -ExpandProperty MediaType"]).strip() == "HDD"
    src = S.out(["findmnt", "-no", "SOURCE", "--target", str(existing(path))]).strip().split("[")[0]
    rota = S.out(["lsblk", "-ndo", "ROTA", src]).split() if src.startswith("/dev/") else []
    return bool(rota) and rota[0] == "1"


def tool_version(exe: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)", S.out([exe, "--version"]))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def venv_tool(name: str):
    p = Path(sys.executable).parent / (name + (".exe" if WIN else ""))
    return str(p) if p.exists() else None


def pick_cmake():
    """CMake 3.24 or newer: the one pip put into .venv first, then the system's."""
    for c in (venv_tool("cmake"), shutil.which("cmake")):
        if c and tool_version(c) >= (3, 24):
            return c
    return None


def choose(question: str, intro: list, options: list, default: int, yes: bool, outro: list = ()) -> int:
    """The answer's index among `options`, each (label, note or None, about or None): a menu on the setup's screen
    (tools/setup_tui.py; `about` shows under the highlighted one), else the numbered list and the number typed."""
    if S.UI is not None and not yes:
        pick = S.UI.choose(question, intro, options, default, outro)
        if pick is not None:
            return pick
    say()
    for line in intro:
        say("  " + line)
    for i, (label, note, _) in enumerate(options, 1):
        say(f"  {i}) {label}" + (f"   ({note})" if note else ""))
    for line in outro:
        say("     " + line)
    return int(ask(question, [str(i) for i in range(1, len(options) + 1)], str(default + 1), yes)) - 1


def gpu_label(g) -> str:
    if g.get("vendor") == "amd":
        return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB, {g['arch']})"
    if g.get("integrated"):   # (DGX Spark / GB10: nvidia-smi has no memory.total - the system RAM, shared)
        return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB unified memory, shared with the CPU)"
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def select_build_backend(backend: str) -> None:
    """Separate build folders and stamps prevent reusing a binary from the other backend."""
    global BUILD, EXE, STAMP
    BUILD = ROOT / ("build-hip" if backend == "hip" else "build")
    EXE = BUILD / ("strata.exe" if WIN else "strata")
    STAMP = BUILD / "MAYA-BUILD.json"


def hip_unified_memory(g) -> bool:
    # gfx1151 is Strix Halo / Gorgon Halo, even when KFD/sysfs has no product name or reports
    # only a tiny firmware VRAM carve-out. This is independent of vram_gb.
    return g.get("arch") == "gfx1151" or bool(g.get("integrated")) or bool(
        re.search(r"Ryzen AI Max|Radeon(?:\s*\(TM\))?\s+80[4-6]\dS", g.get("name", ""), re.I))


def hip_clang(root: Path) -> Path | None:
    """ROCm's clang++: <root>/llvm/bin (a system ROCm on Linux), <root>/lib/llvm/bin (TheRock's wheels, and ROCm 10's
    packages, whose <root>/llvm is a link to it when it exists at all), or <root>/bin (AMD's HIP SDK for Windows)."""
    exe = "clang++.exe" if WIN else "clang++"
    return next((p for p in (root / "llvm" / "bin" / exe, root / "lib" / "llvm" / "bin" / exe, root / "bin" / exe)
                 if p.exists()), None)


def hipblas_lib(root: Path) -> bool:
    """Whether a Linux ROCm root has hipBLAS (lib/, or the distribution's lib/<triplet>/)."""
    return bool(list((root / "lib").glob("libhipblas.so*")) or list((root / "lib").glob("*/libhipblas.so*")))


def linux_rocm_root() -> tuple:
    """(the ROCm root, whether it has the HIP compiler and hipBLAS).  ROCM_PATH when set; else /opt/rocm, and when that
    has no compiler (ROCm 10's packages put each release in /opt/rocm/core-<version> and may leave /opt/rocm itself
    without the llvm / lib links; a versioned install may have no /opt/rocm at all), the newest /opt/rocm/core-<v> and
    then the newest /opt/rocm-<v>."""
    def vkey(p):
        return [int(x) for x in re.findall(r"\d+", p.name)]
    env = os.environ.get("ROCM_PATH")
    cands = [Path(env)] if env else [Path("/opt/rocm")]
    if not env:
        cands += sorted((p for p in Path("/opt/rocm").glob("core-[0-9]*") if p.is_dir()), key=vkey, reverse=True)
        cands += sorted((p for p in Path("/opt").glob("rocm-[0-9]*") if p.is_dir()), key=vkey, reverse=True)
    for c in cands:
        r = c.resolve()
        if hip_clang(r) is not None and hipblas_lib(r):
            return r, True
    return cands[0].resolve(), False


def rocm_sdk() -> str | None:
    """TheRock's rocm-sdk (AMD's ROCm wheels for Windows): ROCM_VENV's, this .venv's, else the one on PATH."""
    venv = os.environ.get("ROCM_VENV")
    for c in (venv and str(Path(venv) / "Scripts" / "rocm-sdk.exe"), venv_tool("rocm-sdk"), shutil.which("rocm-sdk")):
        if c and Path(c).exists():
            return c
    return None


def windows_rocm() -> tuple:
    """(ROCm's root, the folder of its DLLs) on Windows, or (None, None): ROCM_PATH, else TheRock's wheels, else AMD's
    HIP SDK for Windows (HIP_PATH, which its installer sets, else the newest in C:\\Program Files\\AMD\\ROCm)."""
    if os.environ.get("ROCM_PATH"):
        root = Path(os.environ["ROCM_PATH"]).resolve()
        return root, root / "bin"
    sdk = rocm_sdk()
    if sdk is not None:
        # the first `rocm-sdk init` unpacks the compiler, the headers and the libraries (about 3 GB, minutes) and links
        # the device wheels in; later ones only check. Before any `rocm-sdk path`: that would unpack it too, unseen and
        # within S.out's minute, and start over on every run (as llama.cpp's Windows CI does: init, then path)
        say("  AMD's ROCm SDK: rocm-sdk init (the first time it unpacks about 3 GB) ...")
        run([sdk, "init"], check=False)
        path = lambda what: (S.out([sdk, "path", what]).strip().splitlines() or [""])[-1].strip()
        root = path("--root")
        if root:
            return Path(root), Path(path("--bin") or Path(root) / "bin")
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm"
    sdks = sorted(base.glob("[0-9]*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)], reverse=True)
    for root in ([Path(os.environ["HIP_PATH"])] if os.environ.get("HIP_PATH") else []) + sdks:
        if hip_clang(root) is not None:
            return root.resolve(), root.resolve() / "bin"
    return None, None


def rocm_devices_missing(archs) -> list:
    """Of these architectures, those whose library kernels (TheRock's rocm-sdk-device-<arch> wheel) are not in this
    .venv; [] for a ROCm SDK elsewhere (ROCM_PATH, ROCM_VENV, PATH: not checked)."""
    if os.environ.get("ROCM_PATH") or rocm_sdk() is None or rocm_sdk() != venv_tool("rocm-sdk"):
        return []
    from importlib import invalidate_caches, metadata
    invalidate_caches()
    missing = []
    for arch in dict.fromkeys(archs):
        try:
            metadata.version(f"rocm-sdk-device-{arch}")
        except metadata.PackageNotFoundError:
            missing.append(arch)
    return missing


def hip_info(bindir) -> str | None:
    """ROCm's hipInfo (on Windows: lists the GPUs as HIP numbers them, with their architecture)."""
    p = Path(bindir) / "hipInfo.exe" if bindir else None
    return str(p) if p and p.exists() else shutil.which("hipInfo")


THEROCK_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"   # AMD's ROCm for Windows: TheRock's pip wheels
THEROCK_VERSION = "7.14.1"                                     # 7.14: the first with Gorgon Halo (Ryzen AI Max 400)
# The oldest HIP setup builds with: ROCm 10 (HIP 7.15) on Linux, TheRock's 7.14 on Windows.  The ROCm 7.0-7.2 releases
# fail with current drivers - every request's kernel launches (issue #1, a 7.0 pip SDK after a kernel update) or the
# RAM tier's host registration (issue #8, 7.2.1) - and ROCm 10 fixed both.  ABYSS_OLD_ROCM=1: build with one anyway
HIP_MIN = (7, 14)


def hip_new_enough(hipcc: str) -> None:
    """Stops setup on a HIP older than HIP_MIN (ABYSS_OLD_ROCM=1: a warning, and the build goes on)."""
    v = tool_version(hipcc)
    if v >= HIP_MIN:
        return
    have = f"HIP {v[0]}.{v[1]} ({hipcc})" if v != (0, 0) else f"{hipcc} (it did not answer --version)"
    why = ("ROCm 7.0-7.2 fail with current drivers: every request's kernel launches (issue #1) or the RAM tier's "
           "registration (issue #8)")
    if os.environ.get("ABYSS_OLD_ROCM") == "1":
        warn(f"{have} is older than HIP {HIP_MIN[0]}.{HIP_MIN[1]}: {why} - building with it anyway (ABYSS_OLD_ROCM=1)")
        return
    fix = (f"let setup install AMD's ROCm SDK {THEROCK_VERSION} into Maya's .venv (unset ROCM_PATH)" if WIN else
           "install ROCm 10 (AMD's amdrocm10.0 packages, /opt/rocm/core-10.0: docs/amd/README.md) and run setup "
           "again, with ROCM_PATH=/opt/rocm/core-10.0 if an older ROCm is in /opt/rocm")
    fail(f"Maya's HIP port requires HIP {HIP_MIN[0]}.{HIP_MIN[1]} or newer (ROCm 10, TheRock {THEROCK_VERSION}); "
         f"found {have}. {why}", fix + " - ABYSS_OLD_ROCM=1 builds with it anyway")


def msvc_version(vcvars) -> tuple:
    """The MSVC toolset Visual Studio's vcvars64.bat sets up (its Microsoft.VCToolsVersion*.default.txt, else the newest
    VC/Tools/MSVC folder), as (major, minor); (0, 0) when unknown."""
    if not vcvars:
        return (0, 0)
    build = Path(vcvars).parent                      # <VS>/VC/Auxiliary/Build
    names = [f.read_text(errors="replace").strip() for f in sorted(build.glob("Microsoft.VCToolsVersion*.default.txt"))]
    names = names or [p.name for p in (build.parents[1] / "Tools" / "MSVC").glob("14.*")]
    found = [tuple(int(x) for x in m.groups()) for m in (re.match(r"(\d+)\.(\d+)", n) for n in names) if m]
    return max(found, default=(0, 0))


HIP_CMATH_FWD = re.compile(r"^[ \t]*#include <__clang_cuda_math_forward_declares\.h>[ \t]*\r?\n", re.M)
HIP_CMATH_MARK = "// Abyss (issue #43): LLVM 01d3932 - the __device__ declarations come before <cmath>"


def hip_cmath_order(clang, check=False) -> None:
    """Windows: MSVC 14.51's <cmath> (STL update 202604) makes isgreater .. isunordered, isnormal and signbit constexpr,
    which HIP takes as __host__ __device__, and the HIP headers of TheRock's ROCm 7.14.1 then fail to declare their
    __device__ versions ("'isgreater' cannot overload", issues #2, #3, #43).  LLVM fixed it by including
    __clang_cuda_math_forward_declares.h before <cmath> in __clang_hip_runtime_wrapper.h (llvm-project 01d3932,
    2026-06-09; in ROCm 10.1, not 7.14.1): setup makes the same one-line move in the compiler's own copy, once, with
    a backup (<name>.abyss-orig); --check only says so.  ABYSS_NO_ROCM_HEADER_FIX=1 leaves the SDK as it is."""
    v = msvc_version(S.find_vcvars(cuda=False))
    if (0, 0) < v < (14, 50) or clang is None:
        return
    for h in sorted(Path(clang).parent.parent.glob("lib/clang/*/include/__clang_hip_runtime_wrapper.h")):
        try:
            text = h.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        fwd, cmath = HIP_CMATH_FWD.search(text), text.find("#include <cmath>")
        if fwd is None or cmath < 0 or fwd.start() < cmath:          # LLVM's fixed order (or an unknown header)
            continue
        msvc = f"MSVC {v[0]}.{v[1]}" if v != (0, 0) else "Visual Studio's MSVC"
        if check or os.environ.get("ABYSS_NO_ROCM_HEADER_FIX") == "1":
            warn(f"{h} includes <cmath> before the __device__ math declarations: with {msvc} the HIP build can stop "
                 "at \"'isgreater' cannot overload\" (issue #43) - " + ("setup moves them (LLVM's fix)" if check else
                 "left as it is (ABYSS_NO_ROCM_HEADER_FIX=1)"))
            continue
        nl = "\r\n" if "\r\n" in text else "\n"
        moved = HIP_CMATH_FWD.sub("", text, count=1)
        at = moved.find("#include <cmath>")
        moved = moved[:at] + HIP_CMATH_MARK + nl + fwd.group(0).strip() + nl + moved[at:]
        backup = h.with_name(h.name + ".abyss-orig")
        try:
            if not backup.exists():
                shutil.copy2(h, backup)
            h.write_bytes(moved.encode("utf-8"))
        except OSError as e:
            fail(f"cannot update {h} ({e}): with {msvc} ROCm's HIP headers stop the build at \"'isgreater' cannot "
                 "overload\" (issue #43)", "move the line '#include <__clang_cuda_math_forward_declares.h>' to just "
                 "above the first '#include <cmath>' in that file (LLVM's fix, llvm-project 01d3932), or set "
                 "ABYSS_NO_ROCM_HEADER_FIX=1 to try without it")
        ok(f"ROCm's HIP headers: the __device__ math declarations now come before {msvc}'s <cmath> (LLVM's fix for "
           f"\"'isgreater' cannot overload\", issue #43); the original is {backup.name}")


def rocm_pip(archs) -> list:
    """pip's command that puts AMD's ROCm SDK for these GPUs into this Python (.venv): the compiler, the libraries and
    their kernels for each architecture."""
    devices = ",".join(f"device-{x}" for x in sorted(set(archs)))
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--index-url", THEROCK_INDEX,
            f"rocm[libraries,devel,{devices}]=={THEROCK_VERSION}"]


def install_rocm(a, chosen) -> None:
    """Windows: AMD's ROCm SDK with the kernels for these GPUs into Maya's .venv (asked first; nothing goes outside
    this folder), else the command."""
    archs = sorted({g["arch"] for g in chosen})
    cmd = rocm_pip(archs)
    what = f"AMD's ROCm SDK for Windows with the kernels for {', '.join(archs)}"
    if getattr(a, "check", False) or sys.prefix == sys.base_prefix:
        fail(f"{what} is not installed", f"install it into Maya's Python environment: {cmdline(cmd)}"
             f" - then run {ME} again (or set ROCM_PATH / ROCM_VENV to a ROCm SDK {THEROCK_VERSION} or newer that has one)")
    if ask(f"Install {what} ({THEROCK_VERSION}) into .venv now (pip, from repo.amd.com)?",
           ["y", "n"], "y", getattr(a, "yes", False)) != "y":
        fail("Maya's HIP engine is compiled with AMD's ROCm SDK", f"install it with: {cmdline(cmd)}")
    run(cmd)
    if windows_rocm()[0] is None:
        fail("pip installed AMD's ROCm SDK, but rocm-sdk does not answer", f"run {cmdline(cmd)} again and check its "
             "messages")


def check_hip_pc(a, installed=False) -> dict:
    if not WIN and (not sys.platform.startswith("linux") or S.is_wsl()):
        fail("Maya's experimental HIP backend runs on native Linux or Windows")
    if WIN and S.find_vcvars(cuda=False) is None:      # (before ROCm's download: ROCm's clang needs its headers)
        fail("Visual Studio Build Tools (2022 or 2026) are needed: ROCm's compiler uses their headers and libraries",
             "winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add "
             "Microsoft.VisualStudio.Workload.VCTools --includeRecommended\" - or "
             "https://visualstudio.microsoft.com/visual-cpp-build-tools/ with 'Desktop development with C++'")
    root, bindir = windows_rocm() if WIN else (None, None)
    hipinfo = hip_info(bindir) if WIN else None
    found = S.amd_gpus(hipinfo) if WIN else S.amd_gpus()
    usable = [g for g in found if g["arch"] in HIP_ARCHS]
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if g in usable else "not supported by Maya's HIP build"))
    if not usable:
        fail("no supported AMD GPU found", "this port targets RX 7900 XT / XTX (gfx1100), RX 9070 / AI PRO R9700 "
             "(gfx1201), and Strix Halo / Gorgon Halo, Radeon 8050S / 8060S / 8065S (gfx1151); their siblings RX 7800 "
             "XT / 7700 XT (gfx1101), RX 7600 (gfx1102), Radeon 890M (gfx1150) and RX 9060 (gfx1200) build too "
             "(not validated here - please report)")
    if WIN and hipinfo and all(g.get("source") == "registry" for g in found):
        warn(f"ROCm's hipInfo sees no AMD GPU (the list above is Windows'): the engine cannot start until it does - "
             f"update AMD's driver (AMD Software: Adrenalin Edition, https://www.amd.com/en/support), restart, and "
             f"check with {hipinfo}")
    elif WIN and len(found) > 1 and any(g.get("source") == "registry" for g in found):
        warn("these GPU numbers are Windows' order (ROCm's hipInfo was not found): HIP's own can differ")
    if a.gpus:
        # two cards split the layers (each caches the experts of its own half); the larger card goes first, as it
        # takes the bigger first half - the order measured on an R9700 + RX 7900 XT
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()]
        except ValueError:
            fail(f"--gpus takes two GPU numbers as --check shows them, e.g. --gpus 0,1, not {a.gpus!r}")
        if len(want) != 2 or want[0] == want[1]:
            fail("Maya's HIP port runs on one GPU or splits the model across two: --gpu 0, or --gpus 0,1")
        picked = [next((g for g in usable if g["index"] == i), None) for i in want]
        for i, g in zip(want, picked):
            if g is None:
                fail(f"GPU {i} is not a supported AMD card")
        apu = next((g for g in picked if hip_unified_memory(g)), None)
        if apu is not None:
            fail(f"GPU {apu['index']} ({apu['name']}) is an APU with unified memory, which stays single-GPU",
                 "use --gpu N for it, or pick two discrete cards with --gpus")
        chosen = sorted(picked, key=lambda g: -g["vram_gb"])
    else:
        one = next((g for g in usable if g["index"] == a.gpu), None) if a.gpu is not None else max(
            usable, key=lambda g: g["vram_gb"])
        if one is None:
            fail(f"GPU {a.gpu} is not a supported AMD card")
        chosen = [one]
    if WIN:
        # no ROCm SDK, or Maya's without the library kernels (rocm-sdk-device-<arch>) for a chosen GPU: offered once,
        # then checked again with ROCm there (its hipInfo numbers the GPUs as HIP does)
        missing = sorted({g["arch"] for g in chosen}) if root is None else rocm_devices_missing(g["arch"] for g in chosen)
        if missing and not installed and (root is None or not getattr(a, "check", False)):   # (--check: says so below)
            install_rocm(a, chosen)
            return check_hip_pc(a, installed=True)
        if root is None:
            fail("AMD's ROCm SDK for Windows was not found after its install", f"run {ME} again")
        if missing:
            warn(f"the ROCm SDK still has no library kernels for {', '.join(missing)} (rocm-sdk-device-...): hipBLAS "
                 f"fails on that GPU - {cmdline(rocm_pip(g['arch'] for g in chosen))}")
        if hip_clang(root) is None or not (list(root.glob("lib/hipblas*.lib")) or list(bindir.glob("hipblas*.dll"))):
            fail(f"ROCm's HIP compiler and hipBLAS are not in {root}",
                 f"install AMD's ROCm SDK into Maya's .venv: {cmdline(rocm_pip(g['arch'] for g in chosen))} - or set "
                 f"ROCM_PATH to the root of a ROCm SDK {THEROCK_VERSION} or newer for Windows")
        hipcc = next((p for p in (bindir / "hipcc.exe", root / "bin" / "hipcc.exe") if p.exists()), None)
        if hipcc is not None:
            hip_new_enough(str(hipcc))
        hip_cmath_order(hip_clang(root), check=getattr(a, "check", False))
    else:
        root, found = linux_rocm_root()
        if not found:
            fail("ROCm's HIP compiler (llvm/bin/clang++ or lib/llvm/bin/clang++) and hipBLAS are required under "
                 f"{root}", "install ROCm 10, or set ROCM_PATH to its root - ROCm 10's packages: "
                 "/opt/rocm/core-<version> (e.g. /opt/rocm/core-10.0); AMD's pip wheels: the directory rocm-sdk path "
                 "--root prints")
        # hipcc: <root>/bin, or lib/llvm/bin (ROCm 10's packages)
        hipcc = next((p for p in (root / "bin/hipcc", root / "lib/llvm/bin/hipcc") if p.exists()), root / "bin/hipcc")
        hip_new_enough(str(hipcc))
        if not shutil.which("c++"):
            fail("a C++ compiler is needed", "Ubuntu/Debian: sudo apt install build-essential")
    cpu, avx2, avx512 = S.cpu_info()
    if not avx2:
        fail(f"the CPU ({cpu}) needs AVX2 for the expert lane")
    total, avail = mem_gb()
    ok(f"using {' + '.join(gpu_label(g) for g in chosen)}; experimental HIP, "
       f"{'two GPUs (layer split)' if len(chosen) == 2 else 'one GPU'}, text only")
    ok(f"ROCm: {root}; CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'})")
    ok(f"RAM: {total:.0f} GB, {avail:.0f} GB available now")
    if any(hip_unified_memory(g) for g in chosen) and WIN:
        # Windows gives the APU's GPU a fixed carve-out, which it does not count as RAM: the engine sizes the pool
        # from it as on a discrete card (src/core/glm_fast_path.cu, fast_setup)
        vram = sum(g["vram_gb"] for g in chosen)
        ok(f"APU: {vram:.0f} GB of the memory is the GPU's (Windows' share for it): the engine fills it with experts "
           "and keeps the next ones in RAM")
        if vram < total:
            warn(f"the GPU has {vram:.0f} GB and Windows {total:.0f} GB: set Variable Graphics Memory higher (AMD "
                 "Software: Performance > Tuning, or the iGPU memory size in the BIOS), e.g. 96 GB on a 128 GB PC, "
                 "and restart - the experts on the GPU are the fastest")
    elif any(hip_unified_memory(g) for g in chosen):
        ok("APU / unified memory: the engine sizes the GPU expert pool from available system RAM")
    select_build_backend("hip")
    archs = list(HIP_BUILD_ARCHS) + sorted({g["arch"] for g in chosen} - set(HIP_BUILD_ARCHS))
    return {"backend": "hip", "gpus": chosen, "archs": archs, "rocm": str(root),
            **({"rocm_bin": str(bindir)} if WIN else {})}


def nvcc_range(archs) -> tuple:
    """The CUDA toolkit versions that can build for these GPUs: (lowest, first too new or None)."""
    lo = (12, 9) if 121 in archs else (12, 8) if max(archs) >= 100 else (12, 0)  # Blackwell 12.8, GB10 (sm_121) 12.9
    hi = (13, 0) if min(archs) < 75 else None       # CUDA 13 no longer compiles for Volta (sm_70)
    return lo, hi


def find_nvcc(archs, given=None) -> tuple:
    """(nvcc, version) for these GPUs: --nvcc, else the newest toolkit that can build for them (Strata's find_nvcc
    takes the newest of all, but CUDA 13 cannot build for Volta), else the newest found (reported as unfit), else
    (None, None)."""
    def version(c):
        v = re.search(r"release (\d+)\.(\d+)", S.out([c, "--version"]))
        return (int(v.group(1)), int(v.group(2))) if v else None
    if given:
        return given, version(given)
    return pick_nvcc([(c, version(c)) for c in dict.fromkeys(c for c in nvcc_candidates() if c and Path(c).exists())],
                     archs)


def nvcc_candidates() -> list:
    """Every place a CUDA toolkit's nvcc may be (some do not exist): PATH, CUDA_PATH, the default install folders."""
    exe = "nvcc.exe" if WIN else "nvcc"
    cands = [shutil.which("nvcc"), os.environ.get("CUDA_PATH") and str(Path(os.environ["CUDA_PATH"]) / "bin" / exe)]
    if WIN:
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        return cands + [str(p / "bin" / exe) for p in sorted(base.glob("v*"))]
    cands += [str(p / "bin" / exe) for p in sorted(Path("/usr/local").glob("cuda*"))]
    return cands + [str(p / "bin" / exe) for p in sorted(Path("/opt").glob("cuda*"))] + ["/usr/bin/nvcc"]


def pick_nvcc(found, archs) -> tuple:
    """Of [(nvcc, version)], the newest that can build for `archs`, else the newest, else (None, None)."""
    found = sorted([f for f in found if f[1]], key=lambda f: f[1], reverse=True)
    lo, hi = nvcc_range(archs)
    fit = [f for f in found if f[1] >= lo and (hi is None or f[1] < hi)]
    return (fit or found or [(None, None)])[0]


def toolkit_hint(archs) -> str:
    if WIN:
        return ("NVIDIA's CUDA Toolkit 12.8 for Windows (the driver can stay as it is): "
                "https://developer.nvidia.com/cuda-12-8-0-download-archive" +
                (" - Volta (V100) needs 12.x, CUDA 13 dropped it" if min(archs) < 75 else ""))
    if min(archs) < 75:
        return ("Volta (V100) needs CUDA 12.x - Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), "
                "or NVIDIA's cuda-toolkit-12-8 package: https://developer.nvidia.com/cuda-12-8-0-download-archive")
    if max(archs) >= 100:
        return "NVIDIA's cuda-toolkit-12-8 (or newer) package: https://developer.nvidia.com/cuda-downloads"
    return ("Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), or NVIDIA's packages: "
            "https://developer.nvidia.com/cuda-downloads")


def cuda_lib_dirs(nvcc: str) -> list:
    """The toolkit's library folders for the engine's LD_LIBRARY_PATH (none for a distribution's /usr/bin/nvcc); on
    Windows its DLL folders (bin, and bin\\x64 in CUDA 13) for the engine's PATH."""
    if WIN:
        b = Path(nvcc).parent
        return [str(p) for p in (b, b / "x64") if p.is_dir()]
    base = Path(nvcc).resolve().parent.parent
    if base == Path("/usr"):
        return []
    return [str(p) for p in (base / "lib64", base / "targets" / "x86_64-linux" / "lib", base / "targets" / "sbsa-linux" / "lib")
            if p.is_dir()]


# ------------------------------------------------------------------------------------------------ 1. the PC
def choose_gpus(a, found) -> list:
    """The GPUs the model runs on: --gpu / --gpus, else asked when several can share it (all of them recommended)."""
    byid = {g["index"]: g for g in found}
    usable = [g for g in found if int(g["arch"]) >= MIN_CC]
    if a.gpus or a.gpu is not None:
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()] if a.gpus else [a.gpu]
        except ValueError:
            fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,1, not {a.gpus!r}")
        if not 1 <= len(want) <= MAX_GPUS or len(set(want)) != len(want):
            fail(f"Maya runs on one GPU or splits the model's layers across up to {MAX_GPUS}: --gpu 0, or "
                 "--gpus 0,1,2,3 (each number once)")
        for i in want:
            if i not in byid or int(byid[i]["arch"]) < MIN_CC:
                fail(f"GPU {i} cannot be used" + ("" if i in byid else " (not found)"),
                     "use one of: " + ", ".join(str(g["index"]) for g in usable))
        return [byid[i] for i in want]
    if not usable:
        fail("none of the GPUs can run Maya", "it needs an NVIDIA GPU of compute capability 7.0 or newer (V100 and newer)")
    best = sorted(usable, key=lambda g: (-round(g["vram_gb"]), g["index"]))
    if len(best) == 1:
        return best
    opts = ([best[:MAX_GPUS]] if len(best) > 2 else []) + [best[:2], best[:1]]
    labels = []
    for gs in opts:
        if len(gs) > 2:
            labels.append(f"all {len(gs)} together: GPUs " + ", ".join(str(g["index"]) for g in gs) +
                          f" ({sum(g['vram_gb'] for g in gs):.0f} GB of VRAM)")
        elif len(gs) == 2:
            labels.append(f"{gpu_label(gs[0])} + {gpu_label(gs[1])} together")
        else:
            labels.append(f"{gpu_label(gs[0])} only")
    pick = choose("Which GPUs?",
                  ["Maya can run on one GPU, or split the model's layers across several: each then caches the experts "
                   "of its", "own layers, so together they hold more of them (V100s: ~50 tok/s benchmark on two, ~24 "
                   "on one)."],
                  [(label, "recommended" if i == 0 else None, None) for i, label in enumerate(labels)], 0,
                  a.yes or a.check,
                  ["(--gpus picks any others, e.g. --gpus 0,2,5; a model with a draft block drafts tokens on two GPUs "
                   "or more)"] if len(best) > 2 and S.UI is None else [])   # (the plain setup's; not on the screen)
    return opts[pick]


def check_pc(a) -> dict:
    step(1, "checking this PC")
    if a.backend == "hip":
        return check_hip_pc(a)
    if not (WIN or sys.platform.startswith("linux")):
        fail("Project Maya runs on Linux and Windows", "the engine needs an NVIDIA GPU and CUDA")
    if WIN:
        warn("Windows support is new (experimental): Maya is developed and measured on Linux - tell us how it runs")
    if S.is_wsl():
        warn("this is WSL2, which Maya does not support: WSL2's driver pins only about 1 GB of "
             "RAM for the GPU, and Maya's RAM tier pins tens of GB. Use a native Linux install.")
    found = S.gpus()
    if not found:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             ("install the NVIDIA driver (https://www.nvidia.com/drivers)" if WIN else
              "install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install)") + f", restart, and run {ME} again")
    say("  NVIDIA GPUs:")
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if int(g["arch"]) >= MIN_CC else
                                       "too old (Maya needs compute capability 7.0 or newer)"))
    chosen = choose_gpus(a, found)
    archs = sorted({int(g["arch"]) for g in chosen})
    vram = sum(g["vram_gb"] for g in chosen)
    ok("using " + " + ".join(gpu_label(g) for g in chosen) +
       (f": the model's layers are split across the {len(chosen)}" if len(chosen) > 1 else ""))
    if vram < 23:
        say(f"  {vram:.0f} GB of VRAM in total: the engine fills it with the most-used experts and serves the rest from "
            "RAM and the SSD - give it a try (more VRAM is faster; tell us your speed)")
    problems = []                                      # (what is wrong, how to fix it): all of them at once

    nvcc, nv = find_nvcc(archs, a.nvcc)
    lo, hi = nvcc_range(archs)
    want = f"CUDA {lo[0]}.{lo[1]} or newer" + (f" but older than {hi[0]}.0 (CUDA 13 dropped Volta)" if hi else "")
    if nvcc is None:
        problems.append((f"the CUDA toolkit (nvcc) is not installed; these GPUs need {want}", toolkit_hint(archs)))
    elif nv is None:
        problems.append((f"{nvcc} does not say its version (nvcc --version)", "pass the toolkit's nvcc with --nvcc"))
    elif nv < lo or (hi is not None and nv >= hi):
        problems.append((f"nvcc {nv[0]}.{nv[1]} ({nvcc}) cannot build for these GPUs; they need {want}",
                         toolkit_hint(archs) + " (toolkits can be installed side by side: the newest that fits is "
                                                "used, or pass one with --nvcc)"))
    else:
        ok(f"CUDA toolkit {nv[0]}.{nv[1]}: {nvcc}")

    drv = min(S.driver_major(g) for g in chosen)
    need_drv = 580 if nv and nv >= (13, 0) else 525
    if drv < need_drv:
        problems.append((f"the NVIDIA driver {chosen[0]['driver']} is too old: {need_drv} or newer is needed",
                         ("" if WIN else "Ubuntu: sudo ubuntu-drivers install, then restart; or ") +
                         "https://www.nvidia.com/drivers"))
    else:
        ok(f"NVIDIA driver {chosen[0]['driver']}")

    if WIN:
        vcvars = S.find_vcvars()
        if vcvars is None:
            problems.append(("the C++ compiler (Visual Studio 2022 Build Tools, 'Desktop development with C++') is "
                             "not installed", "https://visualstudio.microsoft.com/visual-cpp-build-tools/ - in the "
                             "installer tick 'Desktop development with C++' (CUDA 12 needs 2019 or 2022, not 2026)"))
        else:
            ok(f"C++ compiler: Visual Studio Build Tools ({vcvars})")
    elif a.host_compiler:
        if shutil.which(a.host_compiler) is None:
            problems.append((f"--host-compiler {a.host_compiler} is not installed",
                             f"Ubuntu/Debian: sudo apt-get install -y {Path(a.host_compiler).name}"))
        else:
            ok(f"C++ compiler for nvcc: {a.host_compiler}")
    elif shutil.which("g++") is None:
        problems.append(("the C++ compiler (g++) is not installed", "Ubuntu/Debian: sudo apt-get install -y build-essential"))
    else:
        ok("C++ compiler: " + (S.out(["g++", "--version"]).splitlines() or ["g++"])[0])

    sc = pick_cmake() or shutil.which("cmake")
    scv = tool_version(sc) if sc else (0, 0)
    if scv >= (3, 24):
        ok(f"CMake {scv[0]}.{scv[1]}")
    else:
        ok((f"CMake {scv[0]}.{scv[1]} is older than 3.24" if sc else "no CMake on PATH") +
           ": a current one goes into .venv in step 3 (pip)")
    ok(f"Python {sys.version.split()[0]} ({sys.prefix})")

    total, avail = mem_gb()
    msg = f"RAM: {total:.0f} GB, {avail:.0f} GB available now"
    ok(msg + ("" if total >= 60 else " - it runs with 32 GB; more RAM keeps more experts close and is faster"))
    pf = S.page_file_gb()
    need_pf = int(vram + 8.999)   # the card's memory and the pinned RAM tier are both charged to RAM + page file
    if pf is not None and pf < need_pf:
        warn(f"Windows' page file is {pf:.0f} GB; set it to at least {need_pf} GB. Under Windows every allocation on "
             f"the graphics card ({vram:.0f} GB here) and the engine's pinned RAM tier are both charged to RAM + page "
             f"file, so with {pf:.0f} GB the RAM tier stays about {need_pf - pf:.0f} GB short of your free RAM and "
             "those experts are read from the SSD (issue #20: 2.4-4.1 -> 8.4-11 tokens/s on an RTX 5090). \"System "
             f"managed\" can stay small (8 GB on a 128 GB PC): set a custom size of {need_pf} GB or more - System > "
             "About > Advanced system settings > Performance > Advanced > Virtual memory - and restart")
    cpu, avx2, avx512 = S.cpu_info()
    if platform.machine().lower() in ("aarch64", "arm64"):
        ok(f"CPU: {cpu} (ARM, {os.cpu_count()} threads: no AVX kernels - with an integrated GPU (DGX Spark) the "
           "engine leaves the CPU lane off, the GPU reads the shared memory itself)")
    elif not avx2:
        warn(f"the CPU ({cpu}) has no AVX2: the engine runs, its CPU expert lane on ggml's own kernels (slower; "
             "on 2x Xeon E5-2660 v2 it still added ~15% to 4-GPU decode). STRATA_GLM_CPU_LANE=0 turns it off")
    else:
        ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'}, {os.cpu_count()} threads)")

    if problems:
        say()
        for what, how in problems:
            say(f"  [X]  {what}")
            say(f"       {how}")
        fail("something Maya needs is missing (above)",
             f"install it and run {ME} again - this script installs nothing system-wide")
    select_build_backend("cuda")
    return {"backend": "cuda", "gpus": chosen, "archs": archs, "nvcc": nvcc,
            "vcvars": str(S.find_vcvars()) if WIN else None}


# ------------------------------------------------------------------------------------------------ 2. choices
def choose_context(a, prev_ctx) -> int:
    if a.context:
        return a.context
    default = prev_ctx if prev_ctx in CONTEXTS else DEFAULT_CONTEXT
    pick = choose("Context?", ["Context length = how much text the model sees at once (the chat, files, tool output). "
                               "A longer one takes", "VRAM from the expert cache, so answers get a little slower:"],
                  [(f"{c // 1024}K tokens", "recommended" if c == DEFAULT_CONTEXT else None, None) for c in CONTEXTS],
                  CONTEXTS.index(default), a.yes)
    return CONTEXTS[pick]


def shard_names(m: dict, pattern: str | None = None) -> list:
    """A download's files under their names on Hugging Face: a GGUF's shards, or an EXL3 model's every file."""
    if "files" in m and pattern is None:
        return list(m["files"])
    return [(pattern or m["file"]).format(i=i, n=m["shards"]) for i in range(1, m["shards"] + 1)]


def is_exl3(quant: str) -> bool:
    return MODELS.get(quant, {}).get("format") == "exl3"


def old_names(m: dict) -> list:
    """The names a model's files had on Hugging Face before ("was", newest first), each as its list of shards."""
    return [shard_names(m, p) for p in m.get("was") or []]


def local_shards(m: dict, d: Path) -> list:
    """The model's files in d: under an earlier name when any of its files is there (a download started or finished
    under it - its run config points at them), else under the name on Hugging Face."""
    for names in old_names(m):
        if any((d / n).exists() for n in names):
            return [d / n for n in names]
    return [d / n for n in shard_names(m)]


def hf_name(m: dict, local: Path) -> str:
    """A local file's name on Hugging Face (a file under an earlier name downloads from, and checks against, it)."""
    for names in old_names(m):
        if local.name in names:
            return shard_names(m)[names.index(local.name)]
    return local.name


def download_dir(models: Path, quant: str) -> Path:
    """A download's folder: <models folder>/<quant> (--models-dir).  Files already there are used where they are:
    in that folder, in <models folder>/glm-5.3-flash-<quant> (an earlier version's downloads), or straight in the
    models folder - under their names on Hugging Face or an earlier one."""
    m = MODELS[quant]
    d = models / quant
    for c in (d, models / f"glm-5.3-flash-{quant}".lower(), models):
        for names in [shard_names(m)] + old_names(m):
            if all((c / n).exists() for n in names):
                return c
    return d


def exl3_advice(cards: list, size_gb: float) -> tuple:
    """(recommend, note) for an EXL3 quant of size_gb on this PC.  EXL3 (exllamav3's trellis) is better text prediction
    than a GGUF of its size, but its decode costs more arithmetic: on the GPU's tensor cores (RTX 30 and newer) that is
    cheap, on the RAM tier's lane it is not - an APU's iGPU or an AVX-512 CPU computes an expert in ~0.09-0.12 ms (GLM
    sc_2.10bpw on an RTX 5090 + Radeon 8060S: 22.4 ms a token against Maya-S-v2's 14.8), a CPU with AVX2 only ~20x slower
    than over PCIe.  So: recommended when the cards hold most of its experts, a quality option when the RAM tier has an
    iGPU or AVX-512 to compute on, else not."""
    vram = sum(g.get("vram_gb", 0.0) for g in cards)
    tensor = bool(cards) and all(str(g.get("arch", "0")).isdigit() and int(g.get("arch", "0")) >= 80 for g in cards)
    try:
        avx512 = S.cpu_info()[2]
    except Exception:  # noqa: BLE001
        avx512 = False
    try:
        apu = any(g.get("arch") in ("gfx1150", "gfx1151") for g in S.amd_gpus())
    except Exception:  # noqa: BLE001
        apu = False
    try:
        ram = S.ram_gb()
    except Exception:  # noqa: BLE001
        ram = 0.0
    if ram and vram + 0.75 * ram < size_gb + 8:   # (the RAM tier gets ~3/4 of the RAM; the page cache the rest)
        return False, "experimental - does not fit the GPU and RAM: experts would come from the SSD"
    if tensor and vram >= 0.8 * size_gb:
        return True, "recommended for quality: the cards hold most of its experts"
    if apu or avx512:
        lane = "the iGPU" if apu else "the CPU's AVX-512"
        return False, f"experimental, quality first: better text, slower decode (RAM-tier experts on {lane})"
    return False, "experimental, not for this PC: no AVX-512 or APU iGPU for its RAM-tier experts (much slower)"


def choose_model(a, models: Path, inst: dict) -> tuple:
    """("download", one of MODELS) or ("local", the model's first .gguf file): --gguf-dir, --model, else asked among
    the downloads - the earlier download is the default."""
    if a.gguf_dir:
        first, why = local_first(Path(a.gguf_dir))
        if first is None:
            fail(why, "--gguf-dir takes the folder that holds the GLM-5.3-Flash .gguf files, or the (first) .gguf file")
        return "local", first
    hip = getattr(a, "backend", None) == "hip"
    if a.model:
        if hip and is_exl3(a.model):
            fail(f"{a.model} runs on NVIDIA GPUs only (the HIP port has no EXL3 kernels)",
                 f"pick another model: {ME} --setup")
        return "download", a.model
    opts = [q for q in MODELS if not (hip and is_exl3(q))]   # (EXL3: NVIDIA only)
    # cards of 24 GB or less: Maya-S24 (its 4-bit attention leaves ~1.5 GB more of the card for experts - decode about
    # 14% faster there), else Maya-S
    try:
        cards = [g for g in S.gpus() if g.get("vram_gb")]
    except Exception:  # noqa: BLE001 - no NVIDIA tools (AMD): no card-based recommendation
        cards = []
    rec = "Maya-S24" if cards and all(g["vram_gb"] <= 24.5 for g in cards) and "Maya-S24" in MODELS else opts[0]
    # an EXL3 quant: what this PC makes of it (exl3_advice); the largest one the cards hold is the recommendation then
    advice = {q: exl3_advice(cards, MODELS[q]["download_gb"]) for q in opts if is_exl3(q)}
    fits = [q for q, (r, _) in advice.items() if r]
    if fits:
        rec = max(fits, key=lambda q: MODELS[q]["download_gb"])
    default = inst["quant"] if inst.get("quant") in MODELS else rec
    rows = []
    for q in opts:
        m, d = MODELS[q], download_dir(models, q)
        have = all(p.exists() for p in local_shards(m, d))
        rows.append((f"{q}: " + (f"downloaded, in {d}" if have else f"download {m['download_gb']:.1f} GB from Hugging "
                                                                     "Face"),
                     advice[q][1] if q in advice else
                     ("recommended for cards of 24 GB or less" if q == "Maya-S24" else "recommended") if q == rec
                     else None, m["about"]))
    pick = choose("Model?", ["The model to download (GLM-5.3-Flash GGUF files you already have: --gguf-dir <file or "
                             "folder>):"], rows, opts.index(default), a.yes)
    return "download", opts[pick]


# ------------------------------------------------------------------------------------------------ 3. tools
def tools_step(a) -> Path:
    """The Python packages (into .venv) and llama.cpp's source at the pinned commit - listed, and asked, first."""
    step(3, "Python packages and llama.cpp's source")
    stamp = Path(sys.prefix) / ".strata-pip.json"          # S.pip_install's record of what it installed
    have = read_json(stamp) if stamp.exists() else []
    need = [p for p in PY_PACKAGES if p not in (have if isinstance(have, list) else [])]
    llama = Path(a.llama_dir).expanduser().resolve() if a.llama_dir else ROOT / "third_party" / "llama.cpp"
    llama_ok = (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir()
    if a.llama_dir and not llama_ok:
        fail(f"{llama} is not a llama.cpp checkout (its ggml/ and gguf-py/ folders are missing)")
    downloads = []
    if need:
        downloads.append("Python packages from PyPI into .venv: " + ", ".join(need) + " (roughly 50-100 MB)")
    if not llama_ok:
        downloads.append(f"llama.cpp's source at the pinned commit {S.LLAMA_CPP_COMMIT[:10]} from GitHub, "
                         f"{S.LLAMA_CPP_ZIP} (tens of MB): ggml for the engine build, gguf-py for the pack builder")
    if downloads:
        say("  This step downloads:")
        for d in downloads:
            say("    - " + d)
        if ask("  Download them now?", ["y", "n"], "y", a.yes) != "y":
            fail("nothing was downloaded", f"run {ME} again when you are ready (or pass --llama-dir with a "
                                           f"llama.cpp checkout at commit {S.LLAMA_CPP_COMMIT[:10]})")
    S.pip_install(PY_PACKAGES, ", ".join(PY_PACKAGES))
    if not llama_ok:
        llama = S.get_llama_cpp()
    ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:10]}: {llama}")
    return llama


# ------------------------------------------------------------------------------------------------ 4. the engine
def cached_generator():
    """The CMake generator build/ was configured with before (another one cannot be used there), or None."""
    m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (BUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (BUILD / "CMakeCache.txt").exists() else None
    return m.group(1).strip() if m else None


def arch_setting(archs) -> str:
    """CMAKE_CUDA_ARCHITECTURES for these GPUs.  CMakeLists.txt refuses an explicit arch below 75 (Strata's own
    qwen4exp floor), but the GLM path is developed and measured on Volta V100s (sm_70) - so for those the build
    asks CMake for this machine's own GPUs ("native", which the guard leaves to CMake), with CUDA_VISIBLE_DEVICES
    limited to the chosen cards.  RELEASE-CHECKLIST.md has the one-line CMake fix that makes this unnecessary."""
    return ";".join(str(x) for x in archs) if min(archs) >= 75 else "native"


def compile_engine(archs, gpu_ids, nvcc, host_compiler, llama: Path, src: str, soft=False) -> dict | None:
    """cmake configure + build of the `strata` target in build/; the exact commands are printed as they run.
    soft: a failure warns and returns None (the engine already there keeps working) instead of stopping."""
    def stop(what):
        msg = f"the engine build stopped while {what} (the reason is above)"
        hint = ("common causes: 'unsupported Microsoft Visual Studio version' - CUDA 12 needs Visual Studio 2019 or "
                "2022 (its Build Tools are enough);" if WIN else
                "common causes: 'unsupported GNU version' - install an older g++ your CUDA accepts (e.g. g++-13) and run "
                "again (it is found by itself), or pass --host-compiler g++-13;") + (
                "\n       'Unsupported gpu architecture compute_70' - CUDA 13 cannot build for Volta, "
                "install CUDA 12.x;\n       the compiler killed (out of memory) - close programs and run it again "
                "(it continues)")
        if soft:
            warn(msg + "; starting the engine compiled before")
            return None
        fail(msg, hint)

    cmake = pick_cmake()
    if cmake is None:
        return stop(f"looking for CMake 3.24+ (pip installs one into .venv: run {ME} --setup)")
    gen = cached_generator()
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and gen in (None, "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    cuda_archs = arch_setting(archs)
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}"]
    if host_compiler:
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(host_compiler) or host_compiler}")
    # "native" is what CMake sees: only the chosen cards, numbered as nvidia-smi numbers them
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(str(i) for i in gpu_ids))
    # nvcc takes 2-4 GB per job on the big kernels: half the threads, and no more jobs than RAM / 4 GB
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    build = [cmake, "--build", str(BUILD), "--target", "strata", "abyss_gpu_tune", "-j", str(jobs)]
    say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) +
        " (10-30 minutes the first time, a few minutes after an update) ...")
    if cuda_archs == "native":
        say("  (CMakeLists.txt refuses an explicit sm_70 - Maya's own floor is sm_75 - so CMake is asked for this")
        say(f"  machine's GPUs instead: CMAKE_CUDA_ARCHITECTURES=native with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    conf += S.cmake_fresh(BUILD, conf)   # (another toolkit than build/ was configured with: S.cmake_fresh)
    stopped = cmake_steps(conf, build, env, "build-maya.bat")
    if stopped:
        return stop(stopped)
    meta = {"src": src, "archs": archs, "gpu_ids": list(gpu_ids), "nvcc": nvcc, "host_compiler": host_compiler,
            "llama": str(llama), "lib_dirs": cuda_lib_dirs(nvcc), "date": time.strftime("%Y-%m-%d %H:%M")}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


def cmdline(cmd) -> str:
    """One command as a line of a Windows .bat."""
    return " ".join(f'"{x}"' if re.search(r'[\s;&|<>^()]', str(x)) else str(x) for x in cmd)


def shell_join(cmd) -> str:
    """One command as it is typed here: a POSIX shell, or Windows' cmd."""
    return cmdline(cmd) if WIN else shlex.join(cmd)


def cmake_steps(conf, build, env, bat_name: str, vcvars=None) -> str | None:
    """CMake's configure, then its build (once more when that stops: it continues where it stopped).  None when both
    worked, else what stopped.  On Windows both run in a .bat that first calls Visual Studio's vcvars64.bat - the
    compiler's environment - the way Strata's setup.py builds there (cmake_build)."""
    if not WIN:
        if run(conf, env=env, check=False).returncode != 0:
            return "configuring"
        if run(build, env=env, check=False).returncode != 0:
            say("  (the build stopped - trying it once more: it continues where it stopped)")
            if run(build, env=env, check=False).returncode != 0:
                return "compiling"
        return None
    vcvars = vcvars or S.find_vcvars()
    if vcvars is None:
        return "looking for Visual Studio's C++ compiler (the Build Tools with 'Desktop development with C++')"
    bat = ROOT / bat_name
    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(conf)} || exit /b 3\r\n{cmdline(build)} && exit /b 0'
                   '\r\necho   (the build stopped - trying it once more: it continues where it stopped)\r\n'
                   f'{cmdline(build)} || exit /b 4\r\n', encoding="utf-8")
    say(f"  > {bat}   (Visual Studio's environment, then:)")
    say("  > " + cmdline(conf))
    say("  > " + cmdline(build))
    rc = run(["cmd", "/c", str(bat)], env=env, check=False, echo=False).returncode
    return None if rc == 0 else "compiling" if rc == 4 else "configuring"


PROBE_CU ="""#include <cmath>
#include <string>
#include <vector>
__global__ void k(float* x) { x[0] = rsqrtf(x[0]) + sinf(x[1]) + expf(x[2]) + __expf(x[3]) + sqrtf(x[4]); }
int main() { std::vector<std::string> v(1); return (int) std::cos(0.0) - 1 + (int) v.size() - 1; }
"""


def toolchain_for(archs, nvcc_given: str | None, hc_given: str | None) -> tuple:
    """(nvcc, host compiler or None for the default) that compile the engine's kind of CUDA here.  Every installed
    toolkit that can build for these GPUs (newest first; --nvcc alone if given) with the default g++, then each
    installed g++-N (newest first; --host-compiler alone if given), on a small file the way the engine compiles:
    C++20 and the math functions the kernels use.  A toolkit refuses a g++ newer than it supports, and a newer
    glibc's own rsqrtf/sinpi declarations clash with older CUDA headers - nothing but trying tells which pair works
    on a given system.  On Windows the compiler is Visual Studio's (in its vcvars64.bat environment): only the
    toolkits are tried."""
    import tempfile

    def version(c):
        v = re.search(r"release (\d+)\.(\d+)", S.out([c, "--version"]))
        return (int(v.group(1)), int(v.group(2))) if v else None
    if nvcc_given:
        nvccs = [nvcc_given]
    else:
        seen, found = set(), []
        for c in nvcc_candidates():
            if c and Path(c).exists() and os.path.realpath(c) not in seen:
                seen.add(os.path.realpath(c))
                found.append((c, version(c)))
        lo, hi = nvcc_range(archs)
        nvccs = [c for c, v in sorted((f for f in found if f[1]), key=lambda f: f[1], reverse=True)
                 if v >= lo and (hi is None or v < hi)]
    hcs = [None] if WIN else [hc_given] if hc_given else [None] + sorted(
        {q.name for d in ("/usr/bin", "/usr/local/bin") for q in Path(d).glob("g++-[0-9]*")
         if re.fullmatch(r"g\+\+-\d+", q.name)}, key=lambda n: -int(n[4:]))
    vcvars = S.find_vcvars() if WIN else None
    say("  Finding a CUDA toolkit and C++ compiler that build the engine here ...")
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "probe.cu"
        src.write_text(PROBE_CU, encoding="utf-8")
        for nv in nvccs:
            for c in hcs:
                cmd = [nv, "-std=c++20", "-c", str(src), "-o", str(Path(t) / "probe.o")] + \
                      (["-ccbin", shutil.which(c) or c] if c else [])
                if vcvars:                             # cl.exe is on PATH only inside Visual Studio's environment
                    bat = Path(t) / "probe.bat"
                    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(cmd)}\r\n', encoding="utf-8")
                    cmd = ["cmd", "/c", str(bat)]
                if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
                    ok(f"{nv} with " + ("Visual Studio's C++ compiler" if WIN else c or "the default g++"))
                    return nv, c
    warn("no installed CUDA toolkit and C++ compiler pair compiled a test file: the build will show why (" +
         toolkit_hint(archs) + ")")
    return (nvccs[0] if nvccs else nvcc_given), hc_given


def compile_engine_hip(pc: dict, llama: Path, src: str, soft=False) -> dict | None:
    root = Path(pc["rocm"])
    cmake = pick_cmake()
    if not cmake:
        fail("CMake 3.24 or newer is needed")
    clang = hip_clang(root) or root / "llvm/bin/clang++"
    bindir = Path(pc.get("rocm_bin") or root / "bin")
    env = dict(os.environ, HIP_PLATFORM="amd", HIP_COMPILER="clang", HIP_RUNTIME="rocclr",
               ROCM_PATH=str(root), HIP_PATH=str(root))
    env["PATH"] = os.pathsep.join([str(bindir), str(clang.parent), env.get("PATH", "")])
    if not WIN:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([str(root / "lib"), env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    # Keep compiler-cache writes inside this project, including when run in a workspace sandbox.
    env.setdefault("CCACHE_DIR", str(BUILD / ".ccache"))
    conf = [cmake]
    if WIN:
        # as tools\hip\build_maya_windows.bat, which built it on Windows (#54): ROCm's clang for the host code too, in
        # Visual Studio's environment (cmake_steps), with Ninja, the paths with forward slashes; AVX2 ggml
        ninja = venv_tool("ninja") or shutil.which("ninja")
        if not ninja:
            fail("Ninja is needed to compile the HIP engine on Windows", f"run {ME} --setup again (pip puts it into .venv)")
        bitcode = clang.parent.parent / "amdgcn" / "bitcode"
        env["HIP_DEVICE_LIB_PATH"] = str(bitcode)
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={Path(ninja).as_posix()}",
                 f"-DCMAKE_C_COMPILER={(clang.parent / 'clang.exe').as_posix()}",
                 f"-DCMAKE_CXX_COMPILER={clang.as_posix()}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root.as_posix()}",
                 "-DSTRATA_PORTABLE=ON"]
        if " " not in str(root) + str(bitcode):        # (with a space in them, clang takes HIP_PATH and the above)
            conf.append(f"-DCMAKE_HIP_FLAGS=--rocm-path={root.as_posix()} --rocm-device-lib-path={bitcode.as_posix()}")
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_PREFILL_MMQ=ON",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_HIP_ARCHITECTURES={';'.join(pc['archs'])}",
             f"-DCMAKE_HIP_COMPILER={clang.as_posix()}", f"-DCMAKE_PREFIX_PATH={root.as_posix()}",
             f"-DSTRATA_GGML_DIR={llama.as_posix()}"]
    conf += S.cmake_fresh(BUILD, conf)   # (another ROCm than build-hip/ was configured with: S.cmake_fresh)
    jobs = max(1, min(4, (os.cpu_count() or 4) // 2))
    say("  Compiling Maya for AMD " + ", ".join(pc["archs"]) + " ...")
    if cmake_steps(conf, [cmake, "--build", str(BUILD), "--target", "strata", "abyss_gpu_tune", "-j", str(jobs)], env,
                   "build-maya-hip.bat", vcvars=S.find_vcvars(cuda=False) if WIN else None):
        if soft:
            warn("HIP rebuild failed; starting the previous engine")
            return None
        fail("the HIP engine build failed", "see the compiler output above")
    if WIN:
        # Windows takes a program's DLLs from its own folder, then System32 - where AMD's driver puts its own HIP
        # runtime - and PATH only after: the runtime the engine was built with goes next to it (as llama.cpp's
        # Windows builds do; with the driver's, cudaMemGetInfo failed there). lib_dirs keeps the libraries
        for pattern in ("amdhip64*.dll", "amd_comgr*.dll", "rocm_kpack*.dll"):
            for dll in bindir.glob(pattern):
                try:
                    shutil.copy2(dll, EXE.parent)
                except OSError as e:                   # (in use: an engine still running keeps the one it has)
                    warn(f"could not copy {dll.name} next to the engine ({e})")
    meta = {"backend": "hip", "src": src, "archs": pc["archs"], "rocm": str(root),
            "gpu_ids": [g["index"] for g in pc["gpus"]], "llama": str(llama),
            "lib_dirs": [str(bindir if WIN else root / "lib")], "date": time.strftime("%Y-%m-%d %H:%M")}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


def build_step(a, pc, llama: Path) -> dict:
    step(4, "the engine")
    src = S.source_hash(S.ENGINE_SOURCES)              # src/, include/, third_party/ggml, CMakeLists.txt
    meta = read_json(STAMP)
    if pc.get("backend") == "hip":
        if (EXE.exists() and not a.rebuild and meta.get("backend") == "hip" and meta.get("src") == src
                and meta.get("archs") == pc["archs"] and meta.get("rocm") == pc["rocm"]):
            ok(f"HIP engine already compiled: {EXE}")
            return meta
        return compile_engine_hip(pc, llama, src)
    if (EXE.exists() and not a.rebuild and meta.get("src") == src and set(pc["archs"]) <= set(meta.get("archs", []))
            and (not a.nvcc or a.nvcc == meta.get("nvcc"))
            and (not a.host_compiler or a.host_compiler == meta.get("host_compiler"))):
        ok(f"engine already compiled for " + ", ".join(f"sm_{x}" for x in meta["archs"]) + f": {EXE}")
        return meta
    nvcc, hc = toolchain_for(pc["archs"], a.nvcc, a.host_compiler)
    pc["nvcc"] = nvcc or pc["nvcc"]
    return compile_engine(pc["archs"], [g["index"] for g in pc["gpus"]], pc["nvcc"], hc, llama, src)


def refresh_engine(cfg: dict) -> None:
    """At a start: the engine's source changed since it was compiled (a git pull) - compile what changed."""
    meta = read_json(STAMP)
    if not meta or Path(cfg["exe"]).resolve() != EXE.resolve():
        return
    src = S.source_hash(S.ENGINE_SOURCES)
    if meta.get("src") == src:
        return
    if cfg.get("backend") == "hip":
        root = Path(meta.get("rocm") or "/opt/rocm")
        llama = Path(meta.get("llama") or ROOT / "third_party/llama.cpp")
        if hip_clang(root) is None or not (llama / "ggml/CMakeLists.txt").exists():
            warn("HIP compiler or llama.cpp source is missing; starting the previous engine")
            return
        pc = {"rocm": str(root), "archs": meta["archs"],
              "gpus": [{"index": i} for i in cfg.get("gpu", [0])]}
        if WIN and meta.get("lib_dirs"):
            pc["rocm_bin"] = meta["lib_dirs"][0]
        compile_engine_hip(pc, llama, src, soft=True)
        return
    say("  The engine's source changed since it was compiled (an update): compiling what changed ...")
    nvcc, llama = meta.get("nvcc"), Path(meta.get("llama") or ROOT / "third_party" / "llama.cpp")
    if not nvcc or not Path(nvcc).exists() or not (llama / "ggml" / "CMakeLists.txt").exists():
        warn("the compiler or llama.cpp's source it was built with is gone: starting the engine compiled before "
             f"({ME} --setup --rebuild compiles it again)")
        return
    gpu_ids = meta.get("gpu_ids") or (cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [0])
    compile_engine(meta["archs"], gpu_ids, nvcc, meta.get("host_compiler"), llama, src, soft=True)


# ------------------------------------------------------------------------------------------------ 5. the model
def shard_set(first: Path) -> list:
    """All files of a split GGUF from its first one (<name>-00001-of-0000N.gguf), or just the file."""
    m = SHARD_RE.search(first.name)
    if not m:
        return [first]
    n, stem = int(m.group(2)), first.name[:m.start()]
    return [first.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def gguf_head(path: Path) -> tuple:
    """(general.architecture, tensor count) from the start of a GGUF header: only the key/values before the
    architecture are read (gguf-py writes it first), so a folder of big files is scanned quickly.  ("", 0) when it
    is not a GGUF file or the key does not come early."""
    import struct
    from gguf_reader import GGUF_MAGIC, GGUF_META
    try:
        with open(path, "rb") as f:
            def blob():
                (n,) = struct.unpack("<Q", f.read(8))
                if n > 1 << 20:
                    raise ValueError("not a GGUF string")
                return f.read(n)
            magic, _, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
            if magic != GGUF_MAGIC:
                return "", 0
            for _ in range(min(n_kv, 32)):
                key = blob()
                kind, size = GGUF_META[struct.unpack("<I", f.read(4))[0]]
                if kind == "string":
                    val = blob()
                    if key == b"general.architecture":
                        return val.decode("utf-8", "replace"), n_tensors
                elif kind == "array":
                    _, esize = GGUF_META[struct.unpack("<I", f.read(4))[0]]
                    (count,) = struct.unpack("<Q", f.read(8))
                    if esize is None:                  # an array of strings (the vocabulary) - the key was not early
                        break
                    f.seek(esize * count, 1)
                else:
                    f.seek(size, 1)
    except (OSError, ValueError, KeyError, struct.error):
        pass
    return "", 0


def glm_files(d: Path) -> list:
    """The GLM-5.3-Flash models in the folder `d`, as their first files (-00001-of- for a split one).  A single file
    without tensors (a vocabulary, like the vision folder's) is no model; unsloth's split models keep only metadata
    in their first file."""
    try:
        files = sorted(d.glob("*.gguf"))
    except OSError:
        return []
    found = []
    for f in files:
        m = SHARD_RE.search(f.name)
        if m and int(m.group(1)) != 1:
            continue
        arch, n = gguf_head(f)
        if arch in GLM_ARCHS and (n > 0 or m):
            found.append(f)
    return found


def local_first(p: Path) -> tuple:
    """(the model's first .gguf file, None) from one of its files or the folder that holds it, or (None, why not)."""
    p = p.expanduser().resolve()
    if p.is_file():
        first = shard_set(p)[0]
        arch = gguf_head(first)[0] if first.exists() else ""
        if first.exists() and arch not in GLM_ARCHS:
            return None, f"{first.name} is " + (f"a {arch!r} model, not GLM-5.3-Flash ({GLM_ARCHS[0]})" if arch else
                                                "not a GGUF file")
        return first, None
    found = glm_files(p) if p.is_dir() else []
    if not found:
        return None, (f"no GLM-5.3-Flash GGUF file in {p}" if p.is_dir() else f"{p} does not exist")
    if len(found) > 1:
        return None, f"{p} holds {len(found)} GLM-5.3-Flash models: name the file (" + \
                     ", ".join(f.name for f in found) + ")"
    return found[0], None


def incomplete(path: Path):
    """Why this GGUF file is not whole, or None when it is: as long as its own tensor directory says (the same test
    as Strata's check_shards, without stopping).  Reads the header only."""
    if not path.exists():
        return "missing"
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(path)
    except Exception as e:                             # noqa: BLE001 - a partial header raises anything
        return f"not a whole GGUF file ({e})"
    # a shard of metadata alone (unsloth's first: the tokenizer, no tensors) is whole at its header's end - its writer
    # leaves out the alignment padding no data follows (#83), and the engine reads it so
    need = (g.data_start + max(t.offset + (t.expected_bytes() or 0) for t in g.tensors) if g.tensors
            else g.header_end)
    have = path.stat().st_size
    return None if have >= need else f"short: {have:,} of {need:,} bytes"


def file_incomplete(m: dict, path: Path):
    """Why a download's file is not whole, or None when it is: a GGUF by its own tensor directory (incomplete), an
    EXL3 model's file by its published size."""
    if "files" not in m:
        return incomplete(path)
    if not path.exists():
        return "missing"
    have, want = path.stat().st_size, m["files"][hf_name(m, path)]
    return None if have == want else f"{have:,} of {want:,} bytes"


def quant_of(first: Path) -> str:
    for q, mm in MODELS.items():                       # a Maya download, under its name on Hugging Face or its old one
        if "files" in mm:                              # (an EXL3 download is no .gguf file)
            continue
        if first.name in [names[0] for names in [shard_names(mm)] + old_names(mm)]:
            return q
    m = re.match(r"GLM-5\.3-Flash-(.+?)(-\d{5}-of-\d{5})?\.gguf$", first.name, re.I)
    return m.group(1) if m else first.parent.name


def sha256_ok(path: Path, want) -> bool:
    """`want`: the published sha256, or a list of them - a file published again with only its header changed (v1.0.28:
    the architecture spelled as llama.cpp spells it) keeps the earlier download valid."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest() in [w.lower() for w in ([want] if isinstance(want, str) else want)]


def offer_download(a, quant: str, d: Path, shards: list) -> bool:
    """Shows what would be downloaded - the source, the size, the exact commands - and downloads only after a yes
    (or --download-model).  False: nothing was downloaded."""
    m = MODELS[quant]
    missing = [s for s in shards if file_incomplete(m, s)]
    urls = {s: hf_url(m["repo"], m["revision"], m["folder"], hf_name(m, s)) for s in shards}
    on_disk = sum(s.stat().st_size for s in shards if s.exists()) / 1e9
    remaining = max(0.0, m["download_gb"] - on_disk)
    free = shutil.disk_usage(existing(d)).free / 1e9
    curl = shutil.which("curl")
    cmds = [["mkdir"] + ([] if WIN else ["-p"]) + [str(d)]] + \
        [["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]] for s in missing]
    say(f"  {quant}: {m['about']}.")
    say(f"  Source: https://huggingface.co/{m['repo']}" + (f" (folder {m['folder']}/)" if m["folder"] else "") +
        (f" (branch {m['branch']})" if m.get("branch") else "") + "; the files' own license applies.")
    say(f"  The model is {m['download_gb']:.1f} GB in {len(shards)} file{'s' if len(shards) != 1 else ''}; "
        f"{len(missing)} still to download, about "
        f"{remaining:.0f} GB, into")
    say(f"    {d}   ({free:.0f} GB free there)")
    if free < remaining + 3:
        fail(f"not enough free space in {d}: about {remaining + 3:.0f} GB are needed (the model + its pack)",
             f"free some space, or put the model on another drive: {ME} --setup --models-dir " +
             (r"D:\Maya-models" if WIN else "/path/on/nvme"))
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads experts from these files while it answers - use an "
             "NVMe SSD (--models-dir)")
    say("  The exact commands (resumable: running them again continues an interrupted download):")
    for c in cmds:
        say("    " + shell_join(c))
    if "files" in m:                                   # (--gguf-dir takes GGUF files only)
        say(f"  You can also run them yourself (or download the files any other way into that folder), then run {ME}")
        say("  again.")
    else:
        say("  You can also run them yourself (or download the files any other way into that folder, or pass")
        say(f"  --gguf-dir <folder with the files>), then run {ME} again.")
    if not curl:
        warn("curl is not installed" + ("" if WIN else " (sudo apt-get install -y curl)") + ": if you say yes, the "
             "same URLs are downloaded with Python instead (also resumable)")
    if not a.download_model and ask(f"  Download about {remaining:.0f} GB now with these commands? (y/n)",
                                    ["y", "n"], "n", False) != "y":
        say()
        say(f"Nothing was downloaded. When the files are in place, run {ME} again "
            f"({ME} --download-model downloads them without asking).")
        return False
    d.mkdir(parents=True, exist_ok=True)
    for s in missing:
        if curl:
            cmd = ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]]
            if run(cmd, check=False).returncode != 0:
                fail(f"the download of {s.name} stopped (the reason is above)",
                     f"run {ME} --download-model again: it continues where it stopped")
        else:
            S.download(urls[s], s)
        why = file_incomplete(m, s)
        if why:
            fail(f"{s.name} after the download: {why}", f"delete it and run {ME} --download-model again")
        want = m["sha256"].get(hf_name(m, s))
        if want:
            say(f"  checking {s.name}'s sha256 ...")
            if not sha256_ok(s, want):
                fail(f"{s.name}: the sha256 does not match the published one", "delete it and download it again")
        ok(f"{s.name} downloaded")
    return True


def model_step(a, models: Path, choice: tuple):
    """(model folder, shards, quant name), or None when the files are not there and were not downloaded."""
    step(5, "the model (GLM-5.3-Flash)")
    kind, what = choice
    if kind == "local":
        shards, quant, d = shard_set(what), quant_of(what), what.parent
        for s in shards:
            why = incomplete(s)
            if why:
                fail(f"{s.name}: {why}", f"finish copying or downloading it, then run {ME} again")
        if quant not in MODELS:
            warn(f"{quant}: experimental - only {', '.join(MODELS)} has been measured with Maya (README.md)")
    else:
        quant = what
        m = MODELS[quant]
        d = download_dir(models, quant)
        shards = local_shards(m, d)
        if any(file_incomplete(m, s) for s in shards) and not offer_download(a, quant, d, shards):
            return None
    if is_exl3(quant) and kind != "local":
        conf = read_json(d / "config.json")
        method = (conf.get("quantization_config") or {}).get("quant_method")
        if conf.get("model_type") != "glm5_next" or method != "exl3":
            fail(f"{d / 'config.json'}: a {conf.get('model_type')!r} model quantized as {method!r}, not GLM-5.3-Flash "
                 "in EXL3 (glm5_next, exl3)")
    else:
        from gguf_reader import GGUFFile
        arch = str(GGUFFile(shards[0]).metadata.get("general.architecture", ""))
        if arch not in GLM_ARCHS:
            fail(f"{shards[0].name} is a {arch!r} model, not GLM-5.3-Flash (glm5-next)")
    gb = sum(s.stat().st_size for s in shards) / 1e9
    ok(f"{quant}: {len(shards)} file(s), {gb:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: expect slow answers (the engine reads experts from it) - an NVMe SSD "
             "is strongly recommended")
    return d, shards, quant


# ------------------------------------------------------------------------------------------------ 6. the pack
def pack_source(pack: Path):
    """The first .gguf file a pack indexes (native_experts.txt's header names it), or None (no pack there)."""
    try:
        with open(pack / "native_experts.txt", encoding="utf-8") as f:
            m = re.search(r"absolute offsets in ([^,)]+)", f.readline())
    except OSError:
        return None
    return m.group(1) if m else None


def pack_step(a, d: Path, shards: list, llama: Path, quant: str | None = None) -> Path:
    step(6, "the pack (the engine's index of the model files)")
    if quant is not None and is_exl3(quant):
        return exl3_pack_step(a, d)
    # the engine finds the GGUF files at <pack>/.. (src/core/glm_model.cu, load_pack): the pack lives in their folder,
    # in pack/ - or pack-<model>/ when pack/ indexes another model of the same folder
    pack = d / "pack"
    if pack_source(pack) not in (None, shards[0].name):
        pack = d / f"pack-{quant_of(shards[0])}".lower()
    if not a.repack and pack_source(pack) == shards[0].name and all((pack / f).exists() for f in PACK_FILES):
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written next to the GGUF files (the engine finds them at <pack>/..)",
             "link the .gguf files into a writable folder (" + ("mklink /H DIR\\<file> <file>, for each file" if WIN
                                                                else "mkdir -p DIR && ln -s /path/to/*.gguf DIR/") +
             ") and pass "
             "--gguf-dir DIR")
    say("  Writing the index of every tensor, the small float weights (about 1 GB) and the tokenizer; the experts")
    say("  stay in the GGUF files (a few minutes) ...")
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    # --compat-bf16: the GLM pack stores the projections its kernels read as BF16 (router, indexer, absorbed MLA,
    # KDA gates) in BF16 even where the GGUF quantized them or keeps them F32 (tools/iq_pack.py GLM5NEXT_BF16)
    run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
         "--compat-bf16"], env=env)
    missing = [f for f in PACK_FILES if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    ok(f"pack: {pack}")
    return pack


EXL3_GGUF = "maya-exl3.gguf"                       # what tools/exl3_pack.py writes beside an EXL3 model's files


def exl3_pack_step(a, d: Path) -> Path:
    """An EXL3 model's pack (tools/exl3_pack.py): the engine reads the EXL3 weights in place from the safetensors; the
    pack names them, and a small GGUF beside them holds the unquantized tensors and the tokenizer."""
    pack = d / "pack"
    if not a.repack and pack_source(pack) == EXL3_GGUF and (d / EXL3_GGUF).exists() and \
            all((pack / f).exists() for f in PACK_FILES + ("exl3.txt",)):
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack and {EXL3_GGUF} are written next to the model's files")
    say(f"  Writing {EXL3_GGUF} (the unquantized tensors and the tokenizer, about 2 GB), the index of every tensor,")
    say("  the small float weights (about 1 GB) and the tokenizer; the EXL3 weights stay in the safetensors (a few")
    say("  minutes) ...")
    run([sys.executable, str(ROOT / "tools" / "exl3_pack.py"), "--model", str(d), "--out", str(pack)])
    missing = [f for f in PACK_FILES + ("exl3.txt",) if not (pack / f).exists()] + \
        ([] if (d / EXL3_GGUF).exists() else [f"../{EXL3_GGUF}"])
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    ok(f"pack: {pack}")
    return pack


# ------------------------------------------------------------------------------------------------ 7. images
def fetch(url: str, dst: Path, want: str | None) -> None:
    """One file, resumable (curl -C -, else Python), then its sha256 when one is published."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        if run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dst), url], check=False).returncode != 0:
            fail(f"the download of {dst.name} stopped (the reason is above)", f"run {ME} --setup again: it "
                                                                              "continues where it stopped")
    else:
        S.download(url, dst)
    if want and not sha256_ok(dst, want):
        dst.unlink(missing_ok=True)
        fail(f"{dst.name}: the sha256 does not match the published one (the file was removed)",
             f"run {ME} --setup again to download it again")


def compile_vision(pc, meta, llama: Path, src: str) -> bool:
    """strata-vision: llama.cpp's mtmd at the pinned commit with GLM-5.3-Flash's vision tower added
    (tools/vision/glm5next_patch.py, idempotent), for the same GPUs and toolkit as the engine."""
    cmake = pick_cmake()
    if cmake is None:
        return False
    run([sys.executable, str(ROOT / "tools" / "vision" / "glm5next_patch.py"), str(llama)])
    gen = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (VBUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (VBUILD / "CMakeCache.txt").exists() else None
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and (gen is None or gen.group(1).strip() == "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    archs = meta.get("archs") or pc["archs"]
    conf += ["-S", str(ROOT / "tools" / "vision"), "-B", str(VBUILD), "-DCMAKE_BUILD_TYPE=Release",
             f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=ON",
             "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(str(x) for x in archs),
             f"-DCMAKE_CUDA_COMPILER={meta.get('nvcc') or pc['nvcc']}"]
    if meta.get("host_compiler"):
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(meta['host_compiler']) or meta['host_compiler']}")
    conf += S.cmake_fresh(VBUILD, conf)
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    say("  Compiling the vision encoder (llama.cpp's image library with ggml's CUDA kernels: 10-20 minutes, once) ...")
    if cmake_steps(conf, [cmake, "--build", str(VBUILD), "--target", "strata-vision", "-j", str(jobs)], None,
                   "build-maya-vision.bat"):
        return False
    VSTAMP.write_text(json.dumps({"src": src, "archs": archs, "llama": str(llama),
                                  "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1), encoding="utf-8")
    return VEXE.exists()


def vision_step(a, pc, meta, llama: Path, d: Path, quant: str) -> dict | None:
    """The config's "vision" entry - the image encoder and its files - or None (images off).  The server starts the
    encoder only while a request's new pictures are encoded, in GPU memory the model lends it, and measures on the
    first start how much that is on this GPU (serve/server.py, vision_footprint)."""
    step(7, "images (the vision encoder)")
    if pc.get("backend") == "hip":
        ok("HIP port: text only; vision is not enabled")
        return None
    # a GGUF of your own reads pictures with the same files: the vision tower and the tokenizer are the same in
    # every GLM-5.3-Flash quant
    spec = MODELS.get(quant) or MODELS[next(iter(MODELS))]
    m = spec.get("vision")
    if a.no_vision:
        ok("skipped (--no-vision): the model reads text only")
        return None
    if m is None:
        warn(f"no vision files are published for {quant}: the model reads text only")
        return None
    vd = d / m["folder"]
    files = {k: vd / m[k] for k in ("mmproj", "vocab")}
    missing = [p for p in files.values() if not p.exists() or (m["sha256"].get(p.name) and p.stat().st_size == 0)]
    if missing:
        repo, rev = m.get("repo", spec["repo"]), m.get("revision", spec["revision"])
        src_url = {p: hf_url(repo, rev, m["folder"], p.name) for p in missing}
        say(f"  The vision encoder's files ({m['download_gb']:.2f} GB) from https://huggingface.co/"
            f"{repo} (folder {m['folder']}/), into {vd}:")
        for p in missing:
            say("    " + shell_join(["curl", "-L", "--fail", "-C", "-", "-o", str(p), src_url[p]]))
        if not (a.download_model or a.yes) and ask("  Download them now? (y/n)", ["y", "n"], "y", False) != "y":
            warn(f"not downloaded: the model reads text only ({ME} --setup asks again)")
            return None
        for p in missing:
            fetch(src_url[p], p, m["sha256"].get(p.name))
            ok(f"{p.name} downloaded")
    src = S.source_hash(("tools/vision",))
    vmeta = read_json(VSTAMP)
    if not (VEXE.exists() and not a.rebuild and vmeta.get("src") == src
            and set(meta.get("archs") or pc["archs"]) <= set(vmeta.get("archs", []))):
        if not compile_vision(pc, meta, llama, src):
            warn("the vision encoder did not compile (the reason is above): the model reads text only - "
                 f"{ME} --setup --rebuild tries again")
            return None
    ok(f"vision encoder: {VEXE}")
    return {"exe": str(VEXE), "mmproj": str(files["mmproj"]), "model": str(files["vocab"]), "gpu": True}


# ------------------------------------------------------------------------------------------------ 8. config + start
def parse_env(items) -> dict:
    env = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            fail(f"--env takes KEY=VALUE, e.g. --env STRATA_GLM_RAM_HEADROOM_GB=4, not {it!r}")
        env[k] = v
    return env


def write_run_script(cfg_path: Path, port: int) -> Path:
    """run-<config>.sh (.bat): this config's start through abyss.py - on its screen in a terminal (its loading and
    dashboard), else (--plain, or no terminal: a service) the server's text as it prints."""
    cmd = [sys.executable, str(HERE / "abyss.py"), "--config", str(cfg_path), "--port", str(port)]
    if WIN:
        script = ROOT / f"run-{cfg_path.stem}.bat"
        script.write_text(f'@echo off\r\nrem starts the Project Maya dashboard and API (written by abyss.py; {ME} does '
                          f'the same)\r\ncd /d "{ROOT}" || exit /b 1\r\n{cmdline(cmd)} %*\r\n', encoding="utf-8")
        return script
    script = ROOT / f"run-{cfg_path.stem}.sh"
    script.write_text("#!/bin/sh\n# starts the Project Maya dashboard and API (written by abyss.py; ./abyss.sh does the "
                      "same)\ncd " + shlex.quote(str(ROOT)) + " || exit 1\nexec " + shlex.join(cmd) + ' "$@"\n',
                      encoding="utf-8")
    script.chmod(0o755)
    return script


def write_config(a, pc, meta, pack: Path, quant: str, ctx: int, models: Path, vision: dict | None,
                 choice: tuple | None = None) -> Path:
    step(8, "the configuration and the start script")
    port = a.port or 8080
    # --prefill auto (as Strata's setup writes it): the engine picks its prompt chunk - the largest its expert pool
    # can lend, up to 8192; a number (`32768`) sets the chunk itself
    cfg = {"exe": str(EXE), "args": ["--glm-pack", str(pack), "--max-context", str(ctx), "--prefill", "auto"],
           "cwd": str(ROOT),
           "tokenizer": str(pack / "tokenizer"), "model_name": MODEL_NAME, "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": dict(SAMPLING), "reasoning_effort": EFFORT, "lib_dirs": meta.get("lib_dirs") or [],
           "port": port}
    env = parse_env(a.env)
    if pc.get("backend") == "hip":
        cfg["backend"] = "hip"
        if all(g.get("pci") for g in pc["gpus"]):   # (Linux) the same cards when HIP numbers them otherwise
            cfg["gpu_pci"] = [g["pci"] for g in pc["gpus"]]
        # Leave space for the Linux desktop (on a card with a screen on it: an eGPU beside an APU's desktop has none,
        # and 3 GB are ~380 of DeepSeek V4's experts on an AI PRO R9700). The engine sizes its prompt chunk from
        # the available prompt-memory budget; forcing 256 here severely slows HIP.
        gpus = pc["gpus"]
        # two hardware queues a device, written so that a shell's GPU_MAX_HW_QUEUES=1 (an R9700 guides' idle-power
        # setting) does not reach the engine: AI PRO R9700 DeepSeek V4 prompts 8K / 16K 586-650 / 676-684 tok/s on 1,
        # 866-995 / 999-1,011 on 2, the same 12 W idle (serve/server.py sets 2 when nothing does)
        hip_env = {"STRATA_GLM_RAM_HEADROOM_GB": "16", "GPU_MAX_HW_QUEUES": "2"}
        if any(g.get("display", True) for g in gpus):
            hip_env["STRATA_GLM_RESERVE_MB"] = "3072"
        if len(gpus) == 1:
            hip_env["STRATA_GLM_SPLIT"] = "0"   # two cards: the engine picks the split (and drafts with MTP)
        # Larger prompt sub-batches feed the matrix cores much better (7900 XT: ~250 -> ~410 tok/s); they need
        # a bigger prompt budget, borrowed from the expert pool only while a prompt runs.
        if any(hip_unified_memory(g) for g in gpus):   # an APU is always the only GPU (check_hip_pc)
            total_ram, _ = mem_gb()
            if WIN:     # Windows counts the GPU's carve-out apart from its RAM, and its desktop runs on that GPU
                total_ram += sum(g.get("vram_gb", 0) for g in gpus)
            hip_env.update({"STRATA_GLM_RESERVE_MB": "3072" if WIN else "1024", "STRATA_GLM_PREFILL_SUB": "1024",
                            "STRATA_GLM_PREFILL_MB": "6144" if total_ram >= 96 else "4096"})
        elif min(g.get("vram_gb", 0) for g in gpus) >= 20:
            hip_env.update({"STRATA_GLM_PREFILL_SUB": "1024", "STRATA_GLM_PREFILL_MB": "4096"})
        # The prompt projections' hipBLASLt solutions measured per architecture (tools/hip), one table per card's
        # architecture (':'-separated: each card takes its own). The engine refuses a table made for another
        # hipBLASLt version and keeps plain hipBLAS.
        tables = []
        for arch in dict.fromkeys(g.get("arch", "") for g in gpus):
            found = sorted((ROOT / "tools" / "hip").glob(f"{arch}-glm-hipblaslt-*.txt"))
            if found:
                tables.append(str(found[-1]))
        if tables:
            hip_env["STRATA_HIPBLASLT_TUNING"] = ":".join(tables)
        env = {**hip_env, **env}
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if vision:
        cfg["vision"] = vision
    suffix = "-hip" if pc.get("backend") == "hip" else ""
    cfg_path = ROOT / f"maya-{quant.lower()}{suffix}.json"
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    local = choice[1] if choice and choice[0] == "local" else None   # (choose_model's answer)
    if cfg_path.exists():
        kept = keep_args(read_json(cfg_path).get("args") or [], cfg["args"], ("--prefill",))
        if kept:
            ok("kept from the config before: " + ", ".join(kept))
    for line in prefill_tips(cfg["args"], mem_gb()[0], len(pc["gpus"])):
        ok(line)
    gguf_dir = local.parent if local else Path(a.gguf_dir).expanduser().resolve() if a.gguf_dir else None
    cfg["installer"] = {"models_dir": str(models), "quant": quant, "gguf": str(local) if local else None,
                        "gguf_dir": str(gguf_dir) if gguf_dir else None, "written": time.strftime("%Y-%m-%d %H:%M")}
    cal = saved_calibration(cfg, any_context=True)     # tuned on this PC for this model (and context) before
    if cal is not None:
        tuned = CAL.apply(cfg.get("env") or {}, cal.get("settings") or {})
        tuned.update(env)                              # (an --env given now wins)
        if tuned:
            cfg["env"] = tuned
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else "") +
           (": " + ", ".join(f"{k}={v}" for k, v in cal["settings"].items()) if cal.get("settings") else ""))
        if cal.get("context"):
            ok(f"(tuned at context {cal['context']}: {ME} --calibrate tunes this context too)")
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, port)}")
    return cfg_path


PREFILL_BIG_RAM_GB = 96        # Strata's benchmarks: 32768-token chunks +21-35% at 96 GB, ~3x slower at 32 GB
PREFILL_RISK_RAM_GB = 64       # below this a chunk set above 8192 is warned about


def prefill_tips(args: list, ram: float, gpus: int = 2) -> list:
    """Strata's --prefill recommendations for a split (one GPU's auto already takes up to 32768): text only, nothing in
    the config changes."""
    prefill = args[args.index("--prefill") + 1] if "--prefill" in args[:-1] else None
    if gpus < 2:
        return []
    if prefill is not None and prefill.isdigit() and int(prefill) > 8192 and ram < PREFILL_RISK_RAM_GB:
        return [f"warning: --prefill {prefill} on {ram:.0f} GB of RAM: on a split, 32768-token chunks read prompts "
                "slower than --prefill auto with little RAM (two V100s, 30 GB: 551 against 709 tok/s); they paid off "
                "(+21-35%) with 96 GB in community benchmarks"]
    if prefill == "auto" and ram >= PREFILL_BIG_RAM_GB:
        return [f"tip: with {ram:.0f} GB of RAM, --prefill 32768 in the config's args read prompts 21-35% faster in "
                "community benchmarks; not set, nothing changes"]
    return []


def keep_args(old: list, new: list, flags: tuple) -> list:
    """The engine arguments `flags` as the old config had them (edited by hand, or written by an earlier setup):
    their values replace the new ones, or are added.  `new` is changed in place; what was kept is returned."""
    kept = []
    for flag in flags:
        if flag not in old[:-1]:
            continue
        v = old[old.index(flag) + 1]
        if flag in new[:-1]:
            if new[new.index(flag) + 1] == v:
                continue
            new[new.index(flag) + 1] = v
        else:
            new += [flag, v]
        kept.append(f"{flag} {v}")
    return kept


# ------------------------------------------------------------------------------------------------ calibration
CAL_STORE = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "project-maya" / "calibration.json"


def gpu_link_driver(cfg: dict, found: dict) -> tuple[str, str]:
    """("Gen4 x16", "580.65.06") for the config's GPUs: the PCIe link each card and its slot can do together, and the
    driver.  NVIDIA: nvidia-smi's maximum link (the current one trains down to Gen1 when the card is idle, so it is no
    key); AMD: the narrowest maximum on the way from the card to its root port (sysfs), the amdgpu module's version.
    "?" where there is no card or no tool - the key then says so, and two "?" match each other."""
    sel = list(cfg.get("gpu") or [])
    links, drivers = [], []
    if cfg.get("backend") == "hip":
        for pci in cfg.get("gpu_pci") or []:
            chain = pcie_chain(pci) if not WIN else []
            links.append(f"Gen{min(c[2] for c in chain)} x{min(c[3] for c in chain)}" if chain else "?")
        try:
            drivers.append(Path("/sys/module/amdgpu/version").read_text().strip() or "in-tree")
        except OSError:
            drivers.append("?")
    else:
        got = nvidia_links(sel) if sel and shutil.which("nvidia-smi") else {}
        for i in sel:
            chain = got.get(i)
            links.append(f"Gen{chain[0][2]} x{chain[0][3]}" if chain else "?")
            drivers.append((found.get(i) or {}).get("driver") or "?")
    return " + ".join(links) or "?", " + ".join(sorted(set(drivers))) or "?"


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: these GPUs, this CPU and RAM, the model (its quant), the context, text or
    images (the context's KV cache and the image encoder take VRAM from the expert pool), and - last, so the nearest-
    context search below still compares the rest - the GPUs' PCIe link and driver: a card moved between an x16 slot
    and an OCuLink x4, or a driver update, changes what the tuning measured (Maya #419's idea), and the old result was
    reused as if nothing had changed."""
    found = {g["index"]: g for g in S.gpus() or []}
    sel = [found.get(i) or {} for i in cfg.get("gpu") or []]
    a = cfg.get("args") or []
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a[:-1] else "?"
    quant = (cfg.get("installer") or {}).get("quant") or cfg.get("model_name", "?")
    link, driver = gpu_link_driver(cfg, found)
    return "|".join([" + ".join(g.get("name", "?") for g in sel) or "?",
                     f"{sum(g.get('vram_gb', 0) for g in sel):.0f}GB", S.cpu_info()[0], f"{S.ram_gb():.0f}GB", quant,
                     str(ctx), "images" if cfg.get("vision") else "text", link, driver])


KEY_FIELDS = 9                 # the fields of hardware_key; a key with fewer was written before the link and driver


def load_calibrations() -> dict:
    try:
        store = json.loads(CAL_STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return store if isinstance(store, dict) else {}


_STALE_SAID: set = set()


def stale_calibration_note(store: dict, key: str) -> str | None:
    """One line when the store holds a tuning for this PC and model that is not used because the GPUs' PCIe link or
    driver has changed since (or the entry predates the link and driver in the key); None when there is none.  Said
    once per key."""
    want = key.split("|")
    for k, v in store.items():
        have = k.split("|")
        if not isinstance(v, dict) or have[:5] != want[:5] or have[6:7] != want[6:7] or have[7:] == want[7:]:
            continue
        when = f" ({v['date']})" if v.get("date") else ""
        if len(have) < KEY_FIELDS:
            was = "before the GPU's PCIe link and driver were part of the record"
        else:
            was = f"on {have[7]}, driver {have[8]} - now {want[7]}, driver {want[8]}"
        note = (f"the tuning saved for this PC and model{when} was measured {was}: it is not used, "
                f"{ME} --calibrate measures again")
        if key not in _STALE_SAID:
            _STALE_SAID.add(key)
            warn(note)
        return note
    return None


def saved_calibration(cfg: dict, any_context=False) -> dict | None:
    """The settings an earlier tuning found for this PC, model and context, if any (CUDA only, as the tuning).
    any_context: else the ones tuned at the nearest other context, with that context in "context" - the CPU lane's
    split depends on the PC far more than on the context, and a setup for another context used to drop them (#57: the
    engine's own settings, half the speed on a two-socket DDR3 PC).  A tuning made on another PCIe link or driver (or
    before they were recorded) is stale: not used, said once (stale_calibration_note)."""
    if cfg.get("backend") == "hip":
        return None
    store, key = load_calibrations(), hardware_key(cfg)
    if key in store:
        return store[key]
    want = key.split("|")
    best, best_d = None, None
    if any_context:
        for k, v in store.items():
            have = k.split("|")
            if len(have) != len(want) or have[:5] != want[:5] or have[6:] != want[6:]:
                continue
            try:
                d = abs(math.log(int(have[5]) / int(want[5])))
            except (ValueError, ZeroDivisionError):
                continue
            if isinstance(v, dict) and (best_d is None or d < best_d):
                best, best_d = dict(v, context=int(have[5])), d
    if best is None and any_context:               # (the setup's lookup: nothing usable at any context)
        stale_calibration_note(store, key)
    return best


def busy_gpus(cfg: dict) -> list:
    """The config's GPUs with more than 1 GiB of VRAM in use (another engine on them): [(index, MiB)]."""
    try:
        txt = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    used = {}
    for line in txt.splitlines():
        i, _, m = line.partition(",")
        try:
            used[int(i)] = int(float(m))
        except ValueError:
            pass
    return [(i, used[i]) for i in cfg.get("gpu") or [] if used.get(i, 0) > 1024]


def calibrate_config(cfg_path: Path) -> bool:
    """Tune the CPU lane on this PC (tools/calibrate_glm.py): the result goes into the config's env and into
    CAL_STORE for this PC, model and context (a setup again applies it)."""
    cfg = read_json(cfg_path)
    if cfg.get("backend") == "hip":
        warn("the tuning measures the CUDA engine's CPU lane (NVIDIA GPUs): the HIP port keeps its own settings")
        return False
    busy = busy_gpus(cfg)
    if busy:
        warn("the tuning needs the GPU(s) to itself: " + ", ".join(f"GPU {i} has {m} MiB in use" for i, m in busy) +
             " (a model server still running?) - stop it, then run ./abyss.sh --calibrate")
        return False
    say()
    say("  Tuning Maya for this PC: the output speed is measured with a few splits of the work between the CPU and")
    say("  the PCIe link, and with fewer CPU threads. It takes about 10-15 minutes (the model loads first); the PC is")
    say("  busy meanwhile.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops a setup: the engine's own choices stay
        warn(f"the tuning did not finish ({e}): the engine's own settings stay")
        why = CAL.engine_error(log, since)
        if why:
            say(f"       the engine said: {why}")
        return False
    env = CAL.apply(cfg.get("env") or {}, res["settings"])
    if env:
        cfg["env"] = env
    else:
        cfg.pop("env", None)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    store = load_calibrations()
    store[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                "date": time.strftime("%Y-%m-%d")}
    try:
        CAL_STORE.parent.mkdir(parents=True, exist_ok=True)
        CAL_STORE.write_text(json.dumps(store, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {CAL_STORE} ({e}): {cfg_path.name} has the settings, a setup again does not")
    speed = f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k}={v}" for k, v in res["settings"].items()) + speed)
    else:
        ok("tuned for this PC: the engine's own settings are already the fastest here" + speed)
    return True


# ------------------------------------------------------------------------------------------------ GPU tuning
GPU_TUNE_REPORT = ROOT / "abyss-gpu-tune.txt"
GPU_TUNE_KEYS = ("STRATA_GLM_MV_Q6K_SPLIT",)          # what abyss_gpu_tune may set (src/program/gpu_tune.cpp)


def all_configs() -> list:
    """Every installed config (the setup's maya-*.json and setup-deepseek.sh's abyss-*.json - its engine in "exe"), the
    most recently used first."""
    out = []
    for p in sorted(ROOT.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):   # (most recently used first)
        if p.name.endswith(".shared-settings.json"):
            continue
        c = read_json(p)
        if isinstance(c, dict) and c.get("exe") and c.get("args"):
            out.append(p)
    return out


def gpu_tune(a) -> int:
    """./abyss.sh --gpu-tune: abyss_gpu_tune measures the engine's per-GPU kernel choices on each GPU (no model, about
    a minute a GPU); a choice that is clearly faster than the engine's own goes into the env of the installed configs
    on that GPU (removed again when the engine's own is as fast), and abyss-gpu-tune.txt gets a report to paste into
    a GitHub issue - the engine's table of per-GPU defaults grows from those."""
    cfgs = all_configs()
    backend = a.backend or (read_json(cfgs[0]).get("backend", "cuda") if cfgs else ("hip" if S.amd_gpus() else "cuda"))
    cfgs = [p for p in cfgs if read_json(p).get("backend", "cuda") == backend]
    select_build_backend(backend)
    tool_name = "abyss_gpu_tune.exe" if WIN else "abyss_gpu_tune"
    tool = BUILD / tool_name
    if not tool.exists() and not (BUILD / "CMakeCache.txt").exists():
        # an engine built by hand in another folder (build-cuda, a DGX Spark kit; issue #37 "not built here yet"):
        # the build folder of the engine the configs run
        for p in cfgs:
            d = Path(str(read_json(p).get("exe") or "")).parent
            if str(d) not in ("", ".") and ((d / tool_name).exists() or (d / "CMakeCache.txt").exists()):
                tool = d / tool_name
                break
    build_dir = tool.parent
    if not tool.exists():
        if not (build_dir / "CMakeCache.txt").exists():
            fail(f"the {backend.upper()} engine is not built here yet", f"run {ME} --setup first")
        say("  Compiling the GPU check (abyss_gpu_tune) ...")
        cmake = shutil.which("cmake") or "cmake"
        if subprocess.run([cmake, "--build", str(build_dir), "--target", "abyss_gpu_tune", "-j", str(os.cpu_count() or 4)]
                          ).returncode != 0 or not tool.exists():
            fail("abyss_gpu_tune did not compile", f"{ME} --setup --rebuild compiles the engine again")
    if not WIN and subprocess.run(["pgrep", "-x", "strata"], capture_output=True).returncode == 0:
        fail("an engine is running: it would share the GPU with the measurement", f"stop it first ({ME} stop)")
    if a.gpus or a.gpu is not None:
        gpus = [int(g) for g in str(a.gpus or a.gpu).split(",") if g.strip().isdigit()]
    else:
        gpus = sorted({int(g) for p in cfgs for g in (read_json(p).get("gpu") or [0])}) or [0]
    cfg0 = read_json(cfgs[0]) if cfgs else {}
    env = {k: v for k, v in os.environ.items() if k not in GPU_TUNE_KEYS}   # (the engine's own choice, not an override)
    for k, v in (cfg0.get("env") or {}).items():
        if k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            env[k] = str(v)
    libs = [str(d) for d in cfg0.get("lib_dirs") or []]
    if libs and not WIN:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    say()
    say(f"  Measuring the engine's per-GPU kernel choices on GPU(s) {', '.join(map(str, gpus))} - about a minute each,")
    say("  no model needed.")
    results, text = {}, []
    for g in gpus:
        js = ROOT / f".abyss-gpu-tune-{g}.json"
        say()
        r = subprocess.run([str(tool), "--device", str(g), "--json", str(js)], env=env, capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        for line in (r.stdout or "").splitlines():
            say("    " + line)
        if r.returncode != 0 or not js.exists():
            warn(f"GPU {g}: the measurement failed" + (f" ({out.strip().splitlines()[-1]})" if out.strip() else ""))
            continue
        res = json.loads(js.read_text(encoding="utf-8"))
        js.unlink(missing_ok=True)
        results[g] = res
        text.append(r.stdout.rstrip())
    if not results:
        return 1
    changed = []
    for p in cfgs:
        c = read_json(p)
        on = [int(x) for x in c.get("gpu") or [0]]
        if not all(x in results for x in on):
            continue
        want = {}
        for x in on:                                   # (a split: a choice only when every GPU of it agrees)
            want = results[x]["env"] if x == on[0] else {k: v for k, v in want.items() if results[x]["env"].get(k) == v}
        e = {k: v for k, v in (c.get("env") or {}).items() if k not in GPU_TUNE_KEYS}
        e.update(want)
        if e != (c.get("env") or {}):
            if e:
                c["env"] = e
            else:
                c.pop("env", None)
            p.write_text(json.dumps(c, indent=1), encoding="utf-8")
            changed.append(p.name)
    rep = ["### GPU tune report", "", f"- Abyss {maya_version()}, {backend.upper()}, {S.cpu_info()[0]}, "
           f"{S.ram_gb():.0f} GB RAM", ""]
    for g, res in results.items():
        rep.append(f"- GPU {g}: {res['name']} ({res['arch']}, {res['cus']} CUs, {res['vram_gb']} GB): "
                   + (", ".join(f"{k}={v}" for k, v in res["env"].items()) or "the engine's choices are the fastest"))
    rep += ["", "```", *text, "```", ""]
    try:
        GPU_TUNE_REPORT.write_text("\n".join(rep), encoding="utf-8")
    except OSError:
        pass
    say()
    for g, res in results.items():
        if res["env"]:
            ok(f"GPU {g} ({res['name']}): " + ", ".join(f"{k}={v}" for k, v in res["env"].items()))
        else:
            ok(f"GPU {g} ({res['name']}): the engine's own choices are already the fastest here")
    if changed:
        say("  Written into: " + ", ".join(changed) + " (applies at the next start)")
    say(f"  Report: {GPU_TUNE_REPORT} - please paste it into an issue (https://github.com/zephel01/Abyss/issues) when")
    say("  your GPU is not an AI PRO R9700 or an RTX 5090: the engine's defaults for it come from such reports.")
    return 0


def server_command(cfg_path: Path, a) -> tuple:
    """The dashboard's server for an installed config: (its command, its environment, what the setup's screen shows of
    it).  The engine is compiled again first when its source changed (an update)."""
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    for p, what in ((Path(cfg.get("exe", "")), "the engine"), (pack, "the pack"),
                    (Path(cfg.get("tokenizer", "")) / "vocab.json", "the tokenizer")):
        if p is None or not p.exists():
            fail(f"{cfg_path.name}: {what} is missing ({p})", f"run {ME} --setup to repair it")
    cfg_path.touch()                                   # the most recently used model
    refresh_engine(cfg)
    port = a.port or cfg.get("port") or 8080
    host = a.host or cfg.get("host") or "127.0.0.1"
    key = a.api_key or cfg.get("api_key")
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port)]
    if a.host:
        cmd += ["--host", a.host]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.gpus or a.gpu is not None:                    # this start only, on these cards
        cmd += ["--gpu", str(a.gpus or a.gpu)]
    if (WIN or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not os.environ.get("ABYSS_NO_BROWSER"):
        cmd.append("--open")                           # a desktop: the browser opens when the model is ready
                                                       # (not for Abyss Terminal's detached start: ABYSS_NO_BROWSER)
    here = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    ctx = args[args.index("--max-context") + 1] if "--max-context" in args[:-1] else None
    gpus = str(a.gpus or a.gpu if a.gpus or a.gpu is not None else cfg.get("gpu", 0)).strip("[]").split(",")
    info = {"model": cfg.get("model_name", MODEL_NAME), "quant": (cfg.get("installer") or {}).get("quant"),
            "context": ctx, "dashboard": f"http://{here}:{port}/", "api": f"http://{here}:{port}/v1",
            "log": cfg.get("log"), "key": (key or "").split(",")[0],      # (its dashboard reads /metrics with it)
            "gpus": [int(g) for g in gpus if g.strip().isdigit()]}         # (its load: one part each, in this order)
    env = dict(os.environ, MAYA_RESTART_ON_UPDATE="1")  # (the dashboard may update Maya)
    if S.UI is not None:                               # (the setup's screen shows the same in its card, and reads
        env["PYTHONUNBUFFERED"] = "1"                  # the server's output from a pipe)
        found = S.amd_gpus() if cfg.get("backend") == "hip" else S.gpus()
        info["vram"] = {g["index"]: g["vram_gb"] for g in found}   # (its load: a GPU not started yet, by its VRAM)
        return cmd, env, info
    say()
    say("  " + "-" * 100)
    say(f"  Starting {cfg.get('model_name', MODEL_NAME)} ({cfg_path.name}).")
    say(f"  Dashboard:        http://{here}:{port}/")
    say(f"  API (OpenAI):     http://{here}:{port}/v1        (any model name; the API key: "
        + ("the one you set)" if key else "any)"))
    say(f"  API (Anthropic):  http://{here}:{port}/v1/messages")
    if host not in ("127.0.0.1", "localhost", "::1"):
        say("  Other devices:    the server prints this machine's addresses when it is ready" +
            ("" if key else " - WARNING: no API key, anyone on your network can use it (--api-key KEY)"))
    say("  Loading takes a few minutes: the engine pins up to all but about 6 GB of the free RAM for its expert tier")
    say("  (STRATA_GLM_RAM_HEADROOM_GB changes the 6) and warms its caches - the first answers are the slowest.")
    say("  Ctrl+C (or closing this terminal) stops it. Engine log: " + str(cfg.get("log", "")))
    say("  " + "-" * 100)
    return cmd, env, info


def start(cfg_path: Path, a) -> int:
    """Maya's server for an installed config: on a screen of its own in a terminal (tools/setup_tui.py: its
    dashboard - the Monitor's numbers from the engine - and its log), else in the terminal as it prints."""
    if setup_screen(a, asks=False):
        import setup_tui
        rc = setup_tui.run(lambda: serve_on_screen(cfg_path, a), maya_version(), None, ME, steps=False)
        return after_server(rc, a)
    cmd, env, _ = server_command(cfg_path, a)
    return after_server(subprocess.call(cmd, env=env), a)


def serve_on_screen(cfg_path: Path, a) -> int:
    """Maya's server on the setup's screen (setup.UI): its exit code; one it did not choose stops there, with the
    last line its engine logged in this start (mostly the reason: "strata generate: pack: out of memory")."""
    cmd, env, info = server_command(cfg_path, a)
    log = Path(info["log"]) if info.get("log") else None
    seen = log.stat().st_size if log is not None and log.is_file() else 0
    rc = S.UI.serve(cmd, env, info)
    if rc not in (0, UPDATE_EXIT):
        last = last_line(log, seen)
        fail(f"Maya stopped: its server ended with exit code {rc}",
             (f"its engine's last line: {last} (the whole log: {log})" if last else "the reason is in its output")
             + f"; {ME} starts it again")
    return rc


def last_line(path: Path | None, since: int) -> str:
    """The last line written to `path` after byte `since` ("": none, or no such file)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(since, f.seek(0, os.SEEK_END) - 8192))
            lines = [s.strip() for s in f.read().decode(errors="replace").splitlines() if s.strip()]
    except (OSError, TypeError):
        return ""
    return lines[-1][:200] if lines else ""


def after_server(rc: int, a) -> int:
    """The server ended: its exit code - or About > Updates moved this folder to a new release, and the new version
    starts: its abyss.py compiles what changed in the engine and loads the same model (the most recently used, so no
    question)."""
    if rc != UPDATE_EXIT:
        return rc
    again = [sys.executable, str(HERE / "abyss.py"), "--yes"]
    for flag, v in (("--backend", a.backend), ("--port", a.port), ("--host", a.host), ("--api-key", a.api_key),
                    ("--gpu", a.gpu), ("--gpus", a.gpus), ("--config", getattr(a, "config", None))):
        if v is not None:
            again += [flag, str(v)]
    say()
    say("  Updated from the dashboard: starting the new version ...")
    sys.stdout.flush()
    if WIN:
        return subprocess.call(again)
    os.execv(sys.executable, again)
    return 0


# ------------------------------------------------------------------------------------------------ the report
# the engine log's lines that tell where the time goes: how the model was split across VRAM / RAM / the SSD, the
# prompt path's chunks, and the per-token breakdown ("glm stat": VRAM hits, RAM fetches, disk reads, CPU lane)
REPORT_LINES = re.compile(r"glm fast:|glm prefill: (?:CUDA|HIP)|glm split|glm stat|glm slots|glm prefill: \d|ERR|error|failed|"
                          r"out of memory", re.I)
STAT_DECODE = re.compile(r"glm stat: decode ([\d.]+) ms/tok")


def speed_lines(text) -> list:
    """The log lines REPORT_LINES picks.  A request that only read a prompt (one token out) has no decode to report:
    its "glm stat" line keeps only the prompt part, not a meaningless decode speed (96,000 tokens/s, issue #8)."""
    picked = []
    for x in text:
        if not REPORT_LINES.search(x) or "warming the expert tiers" in x:
            continue
        m = STAT_DECODE.search(x)   # no decode: under 1 ms a token, or no expert touched (one prompt token out)
        if m and (float(m.group(1)) < 1.0 or "vram hit 0.00%" in x) and "| prompt " in x:
            x = "glm stat (prompt only): prompt " + x.split("| prompt ", 1)[1]
        picked.append(x.strip())
    return picked


def hip_gpu_report(rocm: Path, bindir=None) -> str:
    """Static AMD details, with KFD's HIP indices/architectures even when no SMI tool is installed.

    Keep the SMI JSON in its own device order: its DRM indices need not match KFD's HIP indices.
    This also retains driver/VRAM fields across the different versions of the two tools.
    On Windows: ROCm's hipInfo (HIP's indices), else Windows' display adapters.
    """
    import platform
    if WIN:
        exe = hip_info(bindir or rocm / "bin")
        gpus = S.amd_gpus(exe)
        lines = [("hipInfo" if exe else "Windows' display adapters (no hipInfo)") + " (HIP GPU indices):"]
        lines += [f"{gpu_label(g)}; driver {g.get('driver', '?')}" for g in gpus] or ["no AMD GPUs found"]
        lines.append(f"Windows {platform.version()}")
        return "\n".join(lines)
    gpus = S.amd_gpus()
    lines = ["KFD topology (HIP GPU indices):"]
    lines += [f"{gpu_label(g)}; driver {g.get('driver', 'amdgpu')}" for g in gpus]
    if not gpus:
        lines.append("no AMD GPUs found in KFD topology")
    try:
        driver = Path("/sys/module/amdgpu/version").read_text().strip()
    except OSError:
        driver = ""
    lines.append(f"amdgpu driver: {driver or 'in-tree'}, kernel {platform.release()}")
    for name, args in (("rocm-smi", ["--showproductname", "--showmeminfo", "vram", "--showdriverversion", "--json"]),
                       ("amd-smi", ["static", "--json"])):
        exe = shutil.which(name)
        if not exe and (rocm / "bin" / name).is_file():
            exe = str(rocm / "bin" / name)
        if not exe:
            continue
        try:
            data = json.loads(S.out([exe, *args]))
        except ValueError:
            continue
        if isinstance(data, (dict, list)) and data:
            lines += [f"\n{name} (tool GPU indices):", json.dumps(data, indent=2)]
            break
    else:
        lines.append("rocm-smi / amd-smi unavailable or did not answer; using KFD topology")
    return "\n".join(lines)


def rocm_report(rocm: Path) -> str:
    """ROCm's install version file first, then hipcc; reporting never requires the compiler."""
    version = ""
    for name in ("version", "version-dev"):
        try:
            version = (rocm / ".info" / name).read_text().strip()
        except OSError:
            continue
        if version:
            break
    if not version:
        hipcc = next((p for p in (rocm / "bin/hipcc", rocm / "bin/hipcc.exe") if p.is_file()), None)
        if hipcc is not None:
            version = S.out([str(hipcc), "--version"]).strip()
    return f"path {rocm}\nversion {version or 'unknown (ROCm version file / hipcc unavailable)'}"


def report(version: str, backend: str | None = None, cfg_path: Path | None = None) -> int:
    """--report: abyss-report.txt in the Abyss folder - this PC, the installed setup and the engine log's speed lines,
    for a bug or speed report.  Nothing is sent anywhere; the home folder is written as ~ and API keys are left out."""
    home = str(Path.home())
    lines = []
    have = [cfg_path] if cfg_path is not None else all_configs()   # (setup-deepseek.sh's abyss-*.json too)
    cfg = {}
    for p in have:
        candidate = read_json(p)
        if backend is None or candidate.get("backend", "cuda") == backend:
            cfg = candidate
            break
    backend = backend or cfg.get("backend", "cuda")
    select_build_backend(backend)
    stamp = read_json(STAMP)

    def add(title, text=""):
        lines.append(f"## {title}")
        lines.extend(str(text).rstrip().replace(home, "~").splitlines() or ["(nothing)"])
        lines.append("")

    git = S.out(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"]).strip()
    add("Abyss", f"version {version}, commit {git or '?'}, Python {sys.version.split()[0]}, "
                f"{sys.platform}{' (WSL)' if S.is_wsl() else ''}")
    import platform
    add("System", f"{platform.platform()}")
    if backend == "hip":
        rocm = Path((cfg.get("env") or {}).get("ROCM_PATH") or os.environ.get("ROCM_PATH")
                    or stamp.get("rocm") or "/opt/rocm").expanduser().resolve()
        add("GPUs", hip_gpu_report(rocm, (stamp.get("lib_dirs") or [None])[0] if WIN else None))
        add("ROCm", rocm_report(rocm))
    else:
        add("GPUs", S.out(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,"
                                         "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,"
                                         "temperature.gpu", "--format=csv"]) or "nvidia-smi did not answer")
    cpu, avx2, avx512 = S.cpu_info()
    total, avail = mem_gb()
    pf = S.page_file_gb()
    add("CPU and RAM", f"{cpu}, {os.cpu_count()} threads, {'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'}\n"
                       f"RAM {total:.1f} GB, {avail:.1f} GB available now" +
                       (f", page file {pf:.0f} GB" if pf is not None else ""))
    if WIN:
        disks = S.out(["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Format-Table -AutoSize "
                                                              "FriendlyName,MediaType,BusType,Size | Out-String -Width 200"])
    else:
        disks = S.out(["lsblk", "-d", "-o", "NAME,MODEL,ROTA,TRAN,SIZE"])
    add("Disks", disks)
    keys = ("backend", "archs", "rocm", "date") if backend == "hip" else ("archs", "nvcc", "host_compiler", "date")
    add("Engine build", json.dumps({k: stamp.get(k) for k in keys})
        if stamp else "not compiled yet")
    if not have:
        add("Setup", f"no maya-*.json yet: {ME} has not finished a setup here")
    for c in have:
        cfg = read_json(c)
        args = cfg.get("args") or []
        pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
        ctx = args[args.index("--max-context") + 1] if "--max-context" in args[:-1] else "?"
        d = pack.parent if pack else None
        where = ""
        if d is not None and d.exists():
            gb = sum(p.stat().st_size for p in d.glob("*.gguf")) / 1e9
            where = (f"\nmodel folder {d}: {gb:.1f} GB of GGUF, {shutil.disk_usage(d).free / 1e9:.0f} GB free, "
                     f"{'a spinning hard disk' if rotational(d) else 'not a hard disk'}")
        add(f"Setup {c.name}", f"model {(cfg.get('installer') or {}).get('quant', '?')}, context {ctx}, "
                               f"GPUs {cfg.get('gpu')}, images {'on' if cfg.get('vision') else 'off'}, "
                               f"settings {json.dumps(cfg.get('env') or {})}" + where)
        log = Path(cfg.get("log") or c.with_suffix(".log"))
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace").splitlines()
            picked = speed_lines(text)
            add(f"Engine log {log.name}: the speed and memory lines (last 80 of {len(picked)})", "\n".join(picked[-80:]))
            add(f"Engine log {log.name}: the last 25 lines", "\n".join(text[-25:]))
        else:
            add(f"Engine log {log.name}", "not written yet (start Maya once and ask it something)")
    bench_txt = ROOT / "abyss-bench.txt"
    if bench_txt.exists():
        add("Benchmark (abyss-bench.txt, from --bench)", bench_txt.read_text(encoding="utf-8", errors="replace"))
    check_txt = ROOT / "abyss-check.txt"
    if check_txt.exists():
        add("Check (abyss-check.txt, from --check)", check_txt.read_text(encoding="utf-8", errors="replace"))
    out = ROOT / "abyss-report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    ok(f"written: {out}")
    say("  Attach this file to your report (GitHub issue, X, or wherever you are asking). It has this PC's hardware,")
    say("  your Abyss setup and the engine's speed lines - no API key, and your home folder is shown as ~.")
    say("  Best: run it right after a slow answer, so the log has that answer's numbers.")
    return 0


# ------------------------------------------------------------------------------------------------ the benchmark
# the same three questions on every machine (greedy, thinking off), after one warm-up answer, so speeds compare
def first_difference(a: list, b: list) -> int | None:
    """The first position where two token lists part (the shorter one's length when one is the other's start), None
    when they are the same - the --bench drafter check (a greedy answer with and without speculative decoding)."""
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None if len(a) == len(b) else min(len(a), len(b)))


SPEC_EARLY = 12   # the drafter check: parting before this many tokens is looked at closer (late parts: near-ties)
SPEC_TIE = 1.0    # ... and is fine when the token path scores both tokens within this many nats (a near-tie)


BENCH_TOPICS = [
    "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.",
    "Explain how photosynthesis works, step by step, for a high-school student.",
    "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.",
]
PCIE_GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.97, 5: 3.94, 6: 7.88}   # per lane and direction


def plan_share(log_lines) -> float | None:
    """The PCIe share the engine started with, from its CPU-lane line ("of 0..8 RAM-tier experts the CPU takes
    k0 .. k8"): the sweep's share that gives the same plan (k = f - round(share f), as set_pcie_share builds it), else
    the mean over f = 1..8 of (f - k) / f, as Glm5Model::pcie_share computes it; 1.0 with the lane off."""
    for x in log_lines:
        m = re.search(r"of 0\.\.8 RAM-tier experts the CPU takes((?: \d+){9})", x)
        if m:
            ks = [int(v) for v in m.group(1).split()]
            for share in (0.0, 0.25, 0.5, 0.75, 1.0):
                if all(ks[f] == f - math.floor(share * f + 0.5) for f in range(9)):
                    return share
            return sum((f - ks[f]) / f for f in range(1, 9)) / 8.0
        if "CPU lane off" in x:
            return 1.0
    return None


def pcie_chain(pci: str, base: str = "/sys/bus/pci/devices") -> list:
    """[(generation, width, max generation, max width)] of each link from the GPU up to its root port (Linux; an AMD
    card's own switch shows x16 inside it - the real limit is the narrowest link on the way)."""
    out = []
    try:
        d = Path(os.path.realpath(Path(base) / pci))
    except OSError:
        return out
    while (d / "current_link_speed").exists():
        try:
            rd = lambda f: (d / f).read_text().strip()
            gen = lambda s: PCIE_GEN.get(s.split()[0].rstrip("GT/s").strip(), 0)
            out.append((gen(rd("current_link_speed")), int(rd("current_link_width")),
                        gen(rd("max_link_speed")), int(rd("max_link_width"))))
        except (OSError, ValueError):
            break
        d = d.parent
    return out


def pcie_link(chain) -> str:
    """"Gen4 x4 (~7.9 GB/s)" of a pcie_chain: the narrowest link on the way, and what the GPU itself could do."""
    if not chain:
        return "?"
    g = min(c[0] for c in chain)
    w = min(c[1] for c in chain)
    gm, wm = chain[0][2], chain[0][3]
    s = f"Gen{g} x{w} (~{PCIE_LANE_GBS.get(g, 0) * w:.1f} GB/s)"
    return s + (f", the GPU can do Gen{gm} x{wm}" if (gm, wm) != (g, w) else "")


class LinkWatch:
    """The PCIe links of the config's GPUs sampled while the engine works: links train down when idle (ASPM - an
    R9700 sits at Gen1 x4 between requests), so the busiest reading is the one that counts.  AMD on Linux: sysfs, from
    the GPU up to its root port (`pcis`); NVIDIA (Linux and Windows): nvidia-smi's current link (`nvidia` indices)."""

    def __init__(self, pcis=(), nvidia=()):
        import threading
        self.pcis, self.nvidia, self.best, self._stop = list(pcis), list(nvidia), {}, threading.Event()
        self._t = None
        if (self.pcis and not WIN) or (self.nvidia and shutil.which("nvidia-smi")):
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()

    def _keep(self, key, chain):
        cur = lambda c: (min(x[0] for x in c), min(x[1] for x in c))
        if chain and (key not in self.best or cur(chain) > cur(self.best[key])):
            self.best[key] = chain

    def _run(self):
        n = 0
        while not self._stop.wait(0.25):
            if not WIN:
                for p in self.pcis:
                    self._keep(p, pcie_chain(p))
            n += 1
            if self.nvidia and n % 4 == 1:   # (a second apart: nvidia-smi takes ~50 ms)
                for i, chain in nvidia_links(self.nvidia).items():
                    self._keep(f"GPU {i}", chain)

    def stop(self) -> dict:
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)
        return {p: pcie_link(c) for p, c in self.best.items()}


def nvidia_links(indices) -> dict:
    """{index: [(generation, width, max generation, max width)]} from nvidia-smi (the card's own link: an NVIDIA GPU
    has no switch of its own, so a riser's or OCuLink's x4 shows here)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,pcie.link.gen.current,pcie.link.width.current,"
                            "pcie.link.gen.max,pcie.link.width.max", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out = {}
    for line in (r.stdout or "").splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            if len(f) == 5 and int(f[0]) in indices:
                out[int(f[0])] = [(int(f[1]), int(f[2]), int(f[3]), int(f[4]))]
        except ValueError:
            continue
    return out


def bench_links(cfg: dict, SV) -> LinkWatch:
    """The LinkWatch for a config's GPUs: its gpu_pci (setup writes them for AMD), else the KFD topology's addresses of
    its GPU numbers; NVIDIA by nvidia-smi index."""
    if cfg.get("backend") == "hip":
        pcis = cfg.get("gpu_pci") or []
        if not pcis and not WIN:
            nodes = {n["index"]: n["pci"] for n in SV.amd_gpu_nodes()}
            pcis = [nodes[i] for i in SV.gpu_list(cfg) if i in nodes]
        return LinkWatch(pcis=pcis)
    return LinkWatch(nvidia=SV.gpu_list(cfg))


def disk_read_gbs(files, gib: float = 2.0) -> float | None:
    """Sequential read speed (GB/s) of the model's own files, from the disk - the pages are dropped from the page cache
    first (posix_fadvise; Linux), or read unbuffered (Windows).  None where that is not possible."""
    files = [f for f in files if f.exists() and f.stat().st_size > 1 << 28]
    if not files:
        return None
    f = max(files, key=lambda p: p.stat().st_size)
    n = min(int(gib * 2**30), f.stat().st_size // 2)
    if WIN:
        return disk_read_gbs_win(f, f.stat().st_size // 4 // 4096 * 4096, n)
    if not hasattr(os, "posix_fadvise"):
        return None
    try:
        fd = os.open(f, os.O_RDONLY)
    except OSError:
        return None
    try:
        start = f.stat().st_size // 4 // 4096 * 4096
        os.posix_fadvise(fd, start, n, os.POSIX_FADV_DONTNEED)
        t0, got = time.time(), 0
        while got < n:
            b = os.pread(fd, 16 << 20, start + got)
            if not b:
                break
            got += len(b)
        dt = time.time() - t0
        os.posix_fadvise(fd, start, n, os.POSIX_FADV_DONTNEED)   # (do not leave 2 GB of it cached for the engine)
        return got / dt / 1e9 if dt > 0 and got > 0 else None
    except OSError:
        return None
    finally:
        os.close(fd)


def disk_read_gbs_win(f: Path, start: int, n: int) -> float | None:
    """Windows: `n` bytes of `f` from `start`, unbuffered (FILE_FLAG_NO_BUFFERING: past the file cache, into a
    page-aligned buffer, sector-aligned offsets)."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.VirtualAlloc.restype = ctypes.c_void_p
        h = k32.CreateFileW(str(f), 0x80000000, 1, None, 3, 0x20000000 | 0x08000000, None)   # GENERIC_READ, OPEN_EXISTING
        if h in (None, wintypes.HANDLE(-1).value):
            return None
    except (OSError, AttributeError):
        return None
    chunk = 16 << 20
    buf = k32.VirtualAlloc(None, chunk, 0x3000, 0x04)   # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
    try:
        if not buf or not k32.SetFilePointerEx(wintypes.HANDLE(h), ctypes.c_longlong(start), None, 0):
            return None
        got, red = 0, wintypes.DWORD(0)
        t0 = time.time()
        while got < n:
            if not k32.ReadFile(wintypes.HANDLE(h), ctypes.c_void_p(buf), chunk, ctypes.byref(red), None) or not red.value:
                break
            got += red.value
        dt = time.time() - t0
        return got / dt / 1e9 if dt > 0 and got > 0 else None
    finally:
        if buf:
            k32.VirtualFree(ctypes.c_void_p(buf), 0, 0x8000)   # MEM_RELEASE
        k32.CloseHandle(wintypes.HANDLE(h))


def model_key(pack: Path | None) -> tuple:
    """(the model's name - its GGUF set without the shard suffix -, the set's bytes) for bench/ref/<name>.json."""
    if pack is None or not pack.parent.exists():
        return "", 0
    ggufs = sorted(pack.parent.glob("*.gguf"))
    first = [p for p in ggufs if re.search(r"-00001-of-\d{5}\.gguf$", p.name)]
    if first:
        stem = re.sub(r"-00001-of-\d{5}\.gguf$", "", first[0].name)
        ggufs = [p for p in ggufs if re.fullmatch(re.escape(stem) + r"-\d{5}-of-\d{5}\.gguf", p.name)]
    else:
        ggufs = sorted(ggufs, key=lambda p: p.stat().st_size)[-1:]
        stem = ggufs[0].stem if ggufs else ""
    return stem, sum(p.stat().st_size for p in ggufs)


# the settings that change a model's numbers on purpose (lossy speed-ups, the abliteration edit): the quality check
# scores without them - it checks this PC's arithmetic, not those settings (their KL is in docs/MODELS.md)
BENCH_LOSSY = ("STRATA_GLM_TAIL_SKIP", "STRATA_GLM_ABL", "STRATA_GLM_DEFER")


def bench_quality(cfg: dict, pack: Path | None, SV) -> dict | None:
    """The engine scores bench/ref/<model>.json's tokens (teacher-forced, the one-token decode path) against the
    top-k log-probs the same model gave on a known-good run (tools/bench_ref.py): KL near 0 = this PC computes what
    that one did; a broken kernel on this GPU shows up as a large KL (issue #9).  None: no reference for this model."""
    name, size = model_key(pack)
    if not name:
        return None
    # <model>.json, then the other copies' references, <model>.<variant>.json (issue #15: the stock files and an
    # overlay of them share a name): the one made from files of this size
    d = ROOT / "bench" / "ref"
    refs = [p for p in [d / f"{name}.json"] + sorted(d.glob(f"{glob.escape(name)}.*.json")) if p.exists()]
    if not refs:
        return None
    ref = next((p for p in refs if read_json(p).get("gguf_bytes") in (None, size)), refs[0])
    r = read_json(ref)
    topk = ref.with_name(r.get("topk") or f"{name}.topk")
    if r.get("gguf_bytes") not in (None, size) or not topk.exists() or not r.get("ids"):
        return {"skipped": "the reference is for another copy of this model (its files differ)"}
    dense = pack / "dense.bin"
    if r.get("dense_bytes") not in (None, dense.stat().st_size if dense.exists() else 0):
        return {"skipped": "this pack's dense weights differ from the reference's (an overlay pack?)"}
    env = SV.child_env(cfg)
    off = sorted(k for k in env if k.startswith(BENCH_LOSSY))   # (the check is of the model as it ships)
    for k in off:
        env.pop(k)
    env.update(STRATA_GLM_SCORE=str(int(r.get("skip", 0))), STRATA_GLM_SCORE_TOPK=str(topk))
    args = [a for a in SV.engine_args(cfg)]
    if "--max-context" in args[:-1]:
        args[args.index("--max-context") + 1] = str(max(2048, len(r["ids"]) + 64))
    try:
        p = subprocess.run([cfg["exe"]] + args + ["--tokens", ",".join(map(str, r["ids"]))], env=env,
                           cwd=cfg.get("cwd") or str(ROOT), capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"skipped": f"the scoring run did not finish ({e})"}
    m = re.search(r"SCORE (\d+) tokens: nll ([\d.]+) .*?top1 ([\d.]+)%.*?KL ([\d.]+) \(max ([\d.]+)\), same top token "
                  r"([\d.]+)%", p.stdout or "")
    if not m:
        tail = ((p.stderr or "").strip().splitlines() or ["no output"])[-1]
        return {"skipped": f"the scoring run printed no score ({tail[:160]})"}
    kl = float(m.group(4))
    return {"tokens": int(m.group(1)), "nll": float(m.group(2)), "kl": kl, "kl_max": float(m.group(5)),
            "same_top": float(m.group(6)), "ref_nll": r.get("nll"), "ref_made": r.get("made_with", "?"),
            "verdict": "OK" if kl <= 0.02 else "slightly off" if kl <= 0.05 else "OFF", "lossy_off": off,
            "beta": bool(r.get("beta"))}


def sweep_report(shares, own, own_disk, start, now, lane_name, how) -> list:
    """The --deep share sweep's lines: (share, tokens/s) runs, the engine's own share before and after them (own),
    those two runs' SSD reads a token (own_disk, may be empty), the engine's plan as a share (start), the speed to
    compare with when there are no own runs (now), and how to set a share (a format string with {s})."""
    best = max(shares, key=lambda x: x[1])
    lines = ["", "PCIe / CPU-lane share sweep (--deep; 0 = all missed experts on the CPU lane, 1 = all over PCIe;",
             "  every run writes the same answer on a warm GPU pool, so these run faster than the decode answers",
             "  above (new text) - compare the shares with each other; see the note below):"]
    lines += [f"  share {s:.2f}: {v:6.1f} tokens/s" + ("  <- fastest" if (s, v) == best else "") for s, v in shares]
    if own:
        # issue #19: the mean of a still-cold first run and a warm last one made share 0 look 22% faster than the
        # engine's own share, which ran 5% faster than it once warm - a share has to beat the engine's best run
        now = max(own)
        lines.append(f"  the engine's own share{f' (~{start:.2f})' if start is not None else ''}: "
                     f"{' / '.join(f'{v:.1f}' for v in own)} tokens/s, the same answer before and after the sweep")
        if len(own_disk) == 2 and own_disk[0] > max(1.0, 2 * own_disk[1]):
            lines.append(f"  (the first of them still read the SSD {own_disk[0]:.1f} times a token, the last "
                         f"{own_disk[1]:.1f}: the tiers were still filling with this answer's experts, so the shares "
                         "run early in the sweep are slowed by the disk, not by their share)")
        elif len(own) == 2 and abs(own[0] - own[1]) > 0.1 * max(own):
            lines.append("  (its two runs differ by more than 10%: read gaps of that size between the shares as noise)")
    if best[1] > 1.05 * now:
        # issue #14: the sweep runs after the engine's own share has filled the GPU's pool - the experts that go
        # over PCIe are promoted into VRAM, and a lower share promotes fewer.  Set at startup, a lower share keeps
        # the startup's pool much longer (Ornith on DDR4: share 0 34.7 tokens/s in the sweep, 21 from the start
        # vs 24.7 for the engine's own), so this is a lead to try, not a setting to take
        lines.append(f"  share {best[0]:.2f} beats the engine's own choice ({now:.1f} tokens/s) on this warm pool. "
                     f"Set at startup it may not: a lower share fills the GPU's pool more slowly. Try "
                     f"{how.format(s=best[0])} and keep it only if a plain --bench gets faster with it")
    elif max(v for _, v in shares) > 1.25 * min(v for _, v in shares):
        slow = min(shares, key=lambda x: x[1])
        lines.append(f"  the engine's own choice is as fast as the best share; share {slow[0]:.2f} is "
                     f"{best[1] / slow[1]:.1f}x slower - " + ("PCIe is the narrow path here (the lane does the "
                     "missed experts better)" if slow[0] > best[0] else "the lane is the slow path here"))
    else:
        lines.append(f"  the share barely moves the speed: neither PCIe nor the {lane_name} is what limits it")
    return lines


def bench(cfg_path: Path, version: str, deep: bool = False, full: bool = False) -> int:
    """--bench: a standard speed test of the installed model on this PC - decode (writing an answer) on three
    questions and prefill (reading a prompt) at 2k and 8k tokens of this folder's docs - through the engine alone, the
    way the dashboard starts it, then where a token's time goes (the engine's own counters: SSD waits, the CPU lane,
    PCIe fetches, the GPU) and a quality check against a known-good run of the same model when bench/ref has one.
    --deep also times the GPU's timeline and sweeps the PCIe / CPU-lane share (what moves the speed here);
    --full repeats the answers and adds a 32k prompt.  Writes abyss-bench.txt (--report includes it)."""
    import urllib.request
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    if cfg.get("backend") == "hip":
        cfg["exe"] = str(EXE)
    port = cfg.get("port") or 8080
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        fail(f"Abyss is running (port {port}): the benchmark needs the GPU memory it holds",
             f"stop it ({ME} stop, or Ctrl+C in its window), then run --bench again")
    except OSError:
        pass
    sys.path.insert(0, str(ROOT))
    import serve.server as SV                      # the server's own engine command and environment
    import strata_tokenizer as ST
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    tp = Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else ""))
    if not (tp / "vocab.json").exists():
        fail(f"{cfg_path.name}: the tokenizer is missing ({tp})", f"run {ME} --setup to repair it")
    tok = ST.Tokenizer.from_dir(tp)
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) +
                       sorted((ROOT / "bench" / "results").glob("*.md")) if p.exists())
    doc = tok.encode(text)
    lens = [n for n in ((2048, 8192, 32768) if full else (2048, 8192)) if n + 64 <= ctx and n <= len(doc)]
    # the model's own chat template, thinking off (GLM's, DeepSeek's, Qwen's alike); GLM's opening without one
    tpl = None
    if (tp / "chat_template.jinja").exists():
        try:
            from serve.frontend import ChatTemplate
            tpl = ChatTemplate(tp / "chat_template.jinja")
        except Exception:                          # (no jinja2, a template it cannot read: GLM's opening below)
            tpl = None

    def chat(q):
        if tpl is not None:
            try:
                s = tpl.render([{"role": "user", "content": q}], enable_thinking=False)
                if isinstance(s, str) and s:
                    return tok.encode(s, parse_special=True)
            except Exception:
                pass
        return tok.encode("[gMASK]<sop><|user|>\n" + q + "<|assistant|>\n</think>", parse_special=True)

    model_files = sorted(pack.parent.glob("*.gguf")) if pack and pack.parent.exists() else []
    disk = disk_read_gbs(model_files)
    log_path = ROOT / "abyss-bench.log"
    env = SV.child_env(cfg)
    if deep:
        env["STRATA_GLM_GAP_PROF"] = "1"
    import tempfile
    import pc_limits as PL
    route_dir = Path(tempfile.mkdtemp(prefix="abyss-bench-routes-"))
    env["STRATA_GLM_ROUTE_LOG"] = str(route_dir / "route")   # (the VRAM estimate of "What limits this PC")
    step(1, f"benchmark: {cfg_path.name} (loading the model first - a minute or a few)")
    t0 = time.time()
    shares, own = [], []
    n_sweep = 0
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, env=env,
                             cwd=cfg.get("cwd") or str(ROOT), bufsize=1)

        last_toks = []                                      # the last answer's tokens (gen)

        def gen(ids, n_new, keys=""):
            p.stdin.write(f"GEN {n_new} temperature=0 {keys}" + ",".join(map(str, ids)) + "\n")
            p.stdin.flush()
            last_toks.clear()
            while True:
                line = p.stdout.readline()
                if not line:
                    fail("the engine stopped during the benchmark", f"its log: {log_path}")
                if line.startswith("T "):
                    last_toks.append(line.split()[1])
                if line.startswith("DONE") or line.startswith("ERR"):
                    return line.split()

        while True:
            line = p.stdout.readline()
            if not line:
                fail("the engine stopped while loading", f"its log: {log_path}")
            if line.startswith("READY"):
                break
        ok(f"loaded in {time.time() - t0:.0f} s")
        watch = bench_links(cfg, SV)
        gen(chat("Say hello in five languages."), 64)          # warm-up: the first answer after a start is slower
        results = []
        first = None                                        # the first answer's tokens (the drafter check below)
        for q in BENCH_TOPICS * (3 if full else 1):
            f = gen(chat(q), 256)
            if first is None:
                first = list(last_toks)
            if f[0] == "DONE" and float(f[4]) > 0:
                results.append(("decode", q, int(f[1]) / float(f[4]) * 1000.0))
                say(f"  decode  {results[-1][2]:6.1f} tokens/s   {q[:60]}")
        n_dec = len(results)
        # the drafter check: the first answer again token by token (GEN spec=0).  A draft decides how many tokens a
        # verify window checks, never which come, but a window sums in another order than the token path, so a
        # near-tie can still part the two late in an answer; parting within the first SPEC_EARLY tokens means the
        # drafter or the window computes something wrong (2026-10-31: qwen4exp's windows on CUDA lost their experts'
        # weights - the answers parted at tokens 2-29, then looped and lost facts)
        f = gen(chat(BENCH_TOPICS[0]), 256, "spec=0 ")
        spec_diff = None if f[0] != "DONE" or first is None else first_difference(first, last_toks)
        spec_bad = spec_diff is not None and spec_diff < SPEC_EARLY
        tie = ""
        if spec_bad and spec_diff < min(len(first), len(last_toks)):
            # an early part can still be a near-tie: which experts sit in VRAM and which run on the CPU lane moves
            # the sums a little, so the first tokens can flip on both builds alike (2026-10-31: GLM at token 7 on
            # v1.3.86 and v1.3.87, V4.1 at 0).  The token path scores both tokens after the same start: within
            # SPEC_TIE nats is a near-tie; a wrong drafter or window picks a token the model does not rate
            head = chat(BENCH_TOPICS[0]) + [int(t) for t in first[:spec_diff]]
            lps = []
            for t in (first[spec_diff], last_toks[spec_diff]):
                ids = head + [int(t)]
                p.stdin.write(f"LOGP {len(head)} " + ",".join(map(str, ids)) + "\n")
                p.stdin.flush()
                line = p.stdout.readline()
                if not line:
                    fail("the engine stopped during the benchmark", f"its log: {log_path}")
                if line.startswith("LP "):
                    lps.append(float(line.split()[1]))
            if len(lps) == 2:
                gap = abs(lps[0] - lps[1])
                spec_bad = gap >= SPEC_TIE
                tie = f", the two tokens {gap:.2f} nats apart"
        spec_txt = ("not checked" if f[0] != "DONE" or first is None else
                    f"same {len(first)} tokens with and without speculative decoding" if spec_diff is None else
                    f"the same first {spec_diff} of {len(first)} tokens with and without speculative decoding" + tie +
                    (" - TOO EARLY" if spec_bad else
                     " (an early near-tie: fine)" if spec_diff < SPEC_EARLY else " (a late near-tie: fine)"))
        say(f"  drafter check: {spec_txt}")
        if lens:                                            # warm-up: the first prompt after a start is slower
            gen(doc[-lens[0]:], 1)                          # (cold caches, issue #8); its opening is not reused below
        for i, n in enumerate(lens):
            f = gen(doc[i * 997:i * 997 + n], 1)            # different openings: nothing reused between them
            if f[0] == "DONE" and float(f[3]) > 0:
                results.append(("prefill", f"{n} tokens", n / float(f[3]) * 1000.0))
                say(f"  prefill {results[-1][2]:6.0f} tokens/s   a {n}-token prompt")
        if deep:                                            # what moves the speed: the missed experts' PCIe share
            say("  the PCIe / CPU-lane share (0 = every missed expert on the CPU lane, 1 = all copied to the GPU):")
            # the engine's own share before and after the five, under the same conditions (issue #14: the sweep runs
            # last, on warmer tiers and shorter answers than the decode above, so it is compared with these).  Every
            # run writes the same answer, so a first, untimed one brings its experts in first: issue #19's first
            # own-share run still read the SSD 6.8 times a token, the share-0 run after it 0.4
            gen(chat(BENCH_TOPICS[0]), 128)
            for s in (None, 0.0, 0.25, 0.5, 0.75, 1.0, None):
                f = gen(chat(BENCH_TOPICS[0]), 128, "" if s is None else f"pcie_frac={s} ")
                n_sweep += 1
                if f[0] == "DONE" and float(f[4]) > 0:
                    v = int(f[1]) / float(f[4]) * 1000.0
                    if s is None:
                        own.append(v)
                        say(f"    the engine's own share: {v:6.1f} tokens/s")
                    else:
                        shares.append((s, v))
                        say(f"    share {s:.2f}: {v:6.1f} tokens/s")
        links = watch.stop()
        chains = dict(watch.best)
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    stats = speed_lines(log_lines)
    recs = bench_records(log_lines)
    dec_recs = recs[1:1 + n_dec] if len(recs) >= 1 + n_dec else []
    lane_name = log_lane_name(log_lines)
    verdict, evidence = bench_verdict(dec_recs, lane_name)
    quality = None
    if pack is not None:
        say("  quality check: scoring a known text against a known-good run of this model ...")
        quality = bench_quality(cfg, pack, SV)
    dec = [r[2] for r in results if r[0] == "decode"]
    found = S.amd_gpus() if cfg.get("backend") == "hip" else S.gpus()
    if cfg.get("backend") == "hip" and SV.gpu_list(cfg):
        by_index = {g["index"]: g for g in found}
        found = [by_index[i] for i in SV.gpu_list(cfg) if i in by_index]
    gpus = ", ".join(f"{g['name']} {g['vram_gb']:.0f} GB" for g in found) or "?"
    total, _ = mem_gb()
    name, size = model_key(pack)
    pre = {r[1]: r[2] for r in results if r[0] == "prefill"}
    q_txt = ("no reference for this model in bench/ref" if quality is None else
             f"skipped: {quality['skipped']}" if "skipped" in quality else
             f"{quality['verdict']} (KL {quality['kl']:.4f} vs a known-good run, same top token "
             f"{quality['same_top']:.1f}%)")
    uma = bool(found) and all(g.get("integrated") for g in found)   # (DGX Spark / GB10: no PCIe link to its memory)
    if uma:
        links = {k: "unified memory (no PCIe link)" for k in links} or {"GPU": "unified memory (no PCIe link)"}
    summary = (f"Abyss v{version} | {name or cfg_path.stem} | {gpus} | "
               f"{'memory unified' if uma else 'PCIe ' + (', '.join(links.values()) or '?')} | RAM {total:.0f} GB | SSD "
               f"{f'{disk:.1f} GB/s' if disk else '?'} | decode {_mean(dec):.1f} tok/s"
               + (f" | prompt 8k {pre['8192 tokens']:.0f} tok/s" if "8192 tokens" in pre else "")
               + f" | quality {q_txt.split(' (')[0]} | bottleneck: {verdict or '?'}")
    lines = ["### Abyss benchmark", "", summary, "",
             f"- date: {time.strftime('%Y-%m-%d %H:%M')}; setup: {cfg_path.name}, context {ctx}, GPUs {cfg.get('gpu')}"
             f"{', backend ' + cfg.get('backend', 'cuda')}",
             f"- model: {name or '?'} ({size / 1e9:.1f} GB of GGUF)" if size else f"- model: {cfg_path.stem}",
             f"- GPUs: {gpus}; CPU {S.cpu_info()[0]}; RAM {total:.0f} GB; model SSD reads "
             f"{f'{disk:.1f} GB/s' if disk else '? (not measured)'}",
             f"- PCIe while working: " + ("; ".join(f"{k} {v}" for k, v in links.items()) or "? (not readable here: an AMD "
                                                                                        "GPU on Windows, or no GPU found)"),
             "", f"decode (writing the answer): mean {_mean(dec):.1f} tokens/s over {len(dec)} answers of 256 tokens "
                 "(greedy, thinking off)"]
    lines += [f"  {r[2]:6.1f} tokens/s  {r[1]}" for r in results if r[0] == "decode"]
    lines += [f"prefill (reading the prompt): {r[2]:.0f} tokens/s, {r[1]}" for r in results if r[0] == "prefill"]
    lines += ["", "where a token's time goes (the engine's counters, decode answers):"] + [f"- {x}" for x in evidence]
    if verdict:
        what, moves = advice(verdict, cfg.get("backend"))
        lines += [f"- likely bottleneck: {what}", f"- what moves it: {moves}"]
    if shares:
        how = (f"\"STRATA_GLM_PCIE_SHARE\": \"{{s:.2f}}\" in {cfg_path.name}'s \"env\"" if cfg.get("backend") == "hip"
               else "STRATA_GLM_PCIE_SHARE={s:.2f} in the config's \"env\"")
        sweep_recs = recs[-n_sweep:] if n_sweep and len(recs) >= n_sweep else []
        own_disk = [sweep_recs[0]["disk_reads"], sweep_recs[-1]["disk_reads"]] if len(sweep_recs) == 7 else []
        if own:
            now = None
        else:                                                # (no own-share runs: the decode answers above)
            now = _mean(r["ms"] for r in dec_recs)
            now = 1000.0 / now if now > 0 else _mean(dec)
        lines += sweep_report(shares, own, own_disk, plan_share(log_lines), now, lane_name, how)
    # what limits this PC: the answer without the drafter (spec=0) split, and what would change each part
    facts = PL.engine_facts(log_lines, pack)
    routes = PL.read_routes(sorted(route_dir.glob("route*")), facts.get("layers") or 0)
    shutil.rmtree(route_dir, ignore_errors=True)
    plain = recs[1 + n_dec] if len(recs) > 1 + n_dec else None
    lines += ["", "What limits this PC (estimates from this run's measurements; --deep adds the host gaps):"]
    lines += PL.what_limits(plain, facts, next(iter(chains.values()), None), disk, routes,
                            found[0]["vram_gb"] if found else None, lane_name)
    lines += ["", f"quality: {q_txt}", f"drafter check (greedy, the first answer): {spec_txt}"]
    if spec_bad:
        lines.append("  a greedy answer must not change with the drafter: the drafter or the verify window computes "
                     "something the token path does not - please report it")
    if quality and quality.get("lossy_off"):
        lines.append(f"  (scored without this setup's {', '.join(quality['lossy_off'])}: they change the numbers on "
                     "purpose)")
    if quality and "kl" in quality and quality["verdict"] == "slightly off":
        # issue #15: a smaller GPU leaves more experts to the CPU lane, which rounds differently from the GPU (GLM on
        # 16 GB here: KL 0.000 -> 0.025) - expected, not a fault
        lines.append("  slightly off is expected when the CPU lane computes many experts (a GPU of 16 GB or less): it "
                     "rounds differently from the GPU the reference ran on")
    if quality and "kl" in quality and quality["verdict"] == "OFF":
        lines.append("  a large KL means this PC computes something the known-good run did not - please report it")
    if quality and quality.get("beta"):
        # beta support (tools/bench_ref.py --beta): measured on one PC only - the reports are how it gets checked
        lines.append("  this model's support is BETA: it was measured on one PC only - please post this report "
                     "(https://github.com/zephel01/Abyss/issues/new?template=bench.md), whatever the numbers")
    lines += ["", "<details><summary>engine lines</summary>", "", "```"] + \
             [x.replace(str(Path.home()), "~") for x in stats[-40:]] + ["```", "", "</details>"]
    out = ROOT / "abyss-bench.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    say("  " + summary)
    for x in evidence + ([f"likely bottleneck: {BENCH_ADVICE[verdict][0]}"] if verdict else []):
        say("  " + x)
    ok(f"written: {out} - paste it into an issue (https://github.com/zephel01/Abyss/issues) with a speed report")
    return 0


# ------------------------------------------------------------------------------------------------ the check
def os_name() -> str:
    """The OS and its version, no machine name (Linux: the distribution's name too; WSL said so)."""
    import platform
    name = f"{platform.system()} {platform.release()}"
    try:
        name = platform.freedesktop_os_release().get("PRETTY_NAME", "") + f" ({name})"
    except (OSError, AttributeError):
        pass
    if WIN:
        name = f"Windows {platform.version()}"
    return name + (" (WSL)" if S.is_wsl() else "")


def check_gpus(cfg: dict) -> list:
    """The config's GPUs: [{name, vram_gb, ...}] in its order."""
    found = S.amd_gpus() if cfg.get("backend") == "hip" else S.gpus()
    import serve.server as SV
    want = SV.gpu_list(cfg)
    if want:
        by_index = {g["index"]: g for g in found}
        found = [by_index[i] for i in want if i in by_index] or found
    return found


def gpu_driver(cfg: dict) -> str:
    if cfg.get("backend") == "hip":
        v = Path("/opt/rocm/.info/version")
        return "ROCm " + (v.read_text().strip() if v.exists() else "?")
    return (S.out(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip().splitlines()
            or ["?"])[0]


def check_model_name(pack: Path | None, cfg: dict, cfg_path: Path) -> tuple:
    """(the model's name, "folder <its folder's name>, quant <q>, <GB> of GGUF") - names only, no path."""
    name, size = model_key(pack)
    quant = (cfg.get("installer") or {}).get("quant")
    if not quant and pack is not None:
        first = sorted(pack.parent.glob("*.gguf"))
        quant = quant_of(first[0]) if first else None
    folder = pack.parent.name if pack is not None else "?"
    return name or cfg_path.stem, (f"folder {folder}, quant {quant or '?'}" +
                                   (f", {size / 1e9:.1f} GB of GGUF" if size else ""))


def finance_check(cfg_path: Path, version: str) -> int:
    """--check --finance: tools/finance_check.py's items through the model's own server (as --check starts it), every
    answer and grade in abyss-finance.txt.  The only score is "n of N items as the rules say" on synthetic accounts."""
    import urllib.request
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    if cfg.get("backend") == "hip":
        cfg["exe"] = str(EXE)
    port = cfg.get("port") or 8080
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        fail(f"Abyss is running (port {port}): the check needs the GPU memory it holds",
             f"stop it ({ME} stop, or Ctrl+C in its window), then run --check --finance again")
    except OSError:
        pass
    refresh_engine(cfg)
    sys.path.insert(0, str(ROOT))
    import abyss_check as CK
    import finance_check as FC
    step(1, f"finance check: {cfg_path.name} ({len(FC.ITEMS)} items; Abyss must stay stopped while it runs)")
    say("  synthetic accounts, Japanese rules; only what does not depend on the market is graded")
    ccfg = dict(cfg, log=str(ROOT / "abyss-finance.log"))
    ccfg.setdefault("cwd", str(ROOT))
    ccfg.pop("api_key", None)
    ccfg.pop("mcp_servers", None)
    env = dict(os.environ, PYTHONUNBUFFERED="1", ABYSS_NO_BROWSER="1")
    env.pop("STRATA_API_KEY", None)
    srv = CK.Server(ROOT, ccfg, env, ROOT / "abyss-finance-server.log")
    try:
        why = srv.wait_ready(3600)
        if why:
            fail(f"the model did not start: {why}", f"its log: {ROOT / 'abyss-finance.log'}")
        res = FC.run(srv.url, say=say)
    finally:
        srv.stop()
    lines = [f"Abyss v{version} finance check | {cfg_path.name} | {FC.summary(res)}", "",
             "Synthetic accounts and Japanese rules; graded only on what does not depend on the market. Not investment "
             "advice, and not a measure of how good a model is at finance.", ""]
    for r in res:
        lines += [f"## {r['name']}: {'as the rules say' if r['ok'] else 'NOT as the rules say'} - {r['why']}", "",
                  r["answer"].strip(), ""]
    out = ROOT / "abyss-finance.txt"
    out.write_text(CK.scrub("\n".join(lines)), encoding="utf-8")
    ok(f"{FC.summary(res)} - every answer in {out.name}")
    return 0


def model_check(cfg_path: Path, version: str, mode: str) -> int:
    """--check with a model installed: tools/abyss_check.py's stability check of it through its own server (the way
    the dashboard starts it).  Writes abyss-check.txt (--report includes it)."""
    import urllib.request
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    if cfg.get("backend") == "hip":
        cfg["exe"] = str(EXE)
    port = cfg.get("port") or 8080
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        fail(f"Abyss is running (port {port}): the check needs the GPU memory it holds",
             f"stop it ({ME} stop, or Ctrl+C in its window), then run --check again")
    except OSError:
        pass
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    tp = Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else ""))
    for p, what in ((Path(cfg.get("exe", "")), "the engine"), (pack, "the pack"), (tp / "vocab.json", "the tokenizer")):
        if p is None or not p.exists():
            fail(f"{cfg_path.name}: {what} is missing ({p})", f"run {ME} --setup to repair it")
    refresh_engine(cfg)
    sys.path.insert(0, str(ROOT))
    import serve.server as SV
    import strata_tokenizer as ST
    import abyss_check as CK
    tok = ST.Tokenizer.from_dir(tp)
    step(1, f"check: {cfg_path.name} ({mode}; Abyss must stay stopped while it runs)")
    git = S.out(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"]).strip()
    hooks = {"say": say, "ok": ok, "warn": warn, "links": lambda c: bench_links(c, SV), "gpus": check_gpus,
             "disk": lambda pk: disk_read_gbs(sorted(pk.parent.glob("*.gguf")) if pk and pk.parent.exists() else []),
             "model": check_model_name, "driver": gpu_driver, "cpu": lambda: S.cpu_info()[0], "os": os_name,
             "commit": lambda: git or "?", "mem_gb": mem_gb,
             "speed_lines": lambda lines: speed_lines(lines),
             "chars_per_token": lambda t: len(t) / max(1, len(tok.encode(t))),
             "count_tokens": lambda t: len(tok.encode(t)), "encode": tok.encode}
    return CK.run(ROOT, cfg_path, cfg, mode, version, hooks)


# ------------------------------------------------------------------------------------------------ the setup
TUI_WHEELS = ROOT / "third_party" / "wheels"       # the setup's screen: Textual (MIT) and what it needs, pure Python


def setup_screen(a, asks: bool = True) -> bool:
    """Whether the setup - or Maya, `asks` False: nothing to answer - runs on a screen of its own (tools/setup_tui.py):
    in a terminal, not with --plain (nor a setup with --yes: nobody there to answer).  Its library comes with Maya
    (TUI_WHEELS) and goes into .venv from there the first time: nothing is downloaded.  False: plain text, as before."""
    if getattr(a, "plain", False) or (asks and getattr(a, "yes", False)) or os.environ.get("TERM") == "dumb" or \
            not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    wheel = max(TUI_WHEELS.glob("textual-*.whl"), default=None)
    if wheel is None:
        return False
    want = wheel.name.split("-")[1]
    from importlib import invalidate_caches, metadata
    try:
        if metadata.version("textual") == want:
            return True
    except metadata.PackageNotFoundError:
        pass
    if sys.prefix == sys.base_prefix:                  # not Maya's .venv: nothing goes into a system Python
        return False
    say("  Preparing the setup's screen (Textual, from third_party/wheels: nothing is downloaded) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-index",
                        "--find-links", str(TUI_WHEELS), f"textual=={want}"], capture_output=True, text=True)
    if r.returncode != 0:
        why = (r.stderr or r.stdout).strip().splitlines()
        warn("the setup's screen could not be installed, so the setup runs in plain text" +
             (f" ({why[-1]})" if why else ""))
        return False
    invalidate_caches()
    return True


def terminal(a) -> int:
    """./abyss.sh --terminal: Abyss Terminal (tools/abyss_terminal.py) - on its Textual screen (installed from
    third_party/wheels the first time, as the setup's), else in plain text."""
    import abyss_terminal
    argv = []
    for flag, val in (("--url", a.url), ("--key-env", a.key_env), ("--config", a.config), ("--demo", a.demo),
                      ("--layout", a.layout)):
        if val is not None:
            argv += [flag, str(val)]
    argv += ["--interval", str(a.interval)]
    argv += [f for f, on in (("--plain", a.plain), ("--once", a.once), ("--json", a.json)) if on]
    screen = not (a.plain or a.once or a.json) and setup_screen(a, asks=False)
    return abyss_terminal.main(argv, textual=screen)


def set_up(a, prev: dict) -> Path | None:
    """Steps 1-8, then the tuning: the config written, or None when the model was not downloaded.  `prev`: the
    config of the setup before (its answers are the defaults)."""
    inst = prev.get("installer") or {}
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None
    pc = check_pc(a)                                   # 1
    step(2, "your choices")                            # 2
    ctx = choose_context(a, prev_ctx)
    ok(f"context: {ctx} tokens")
    # the models folder: --models-dir, else the one set up before (a config from before the option has its data
    # folder, whose models/ held the downloads), else Maya-data/models next to this folder
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else
              Path(inst["data_dir"]) / "models" if inst.get("data_dir") else ROOT.parent / "Maya-data" / "models")
    choice = choose_model(a, models, inst)
    if choice[0] == "download":
        ok(f"model: {choice[1]}, in {download_dir(models, choice[1])}")
    else:
        ok(f"model: {choice[1]} (yours)")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama)                    # 4
    got = model_step(a, models, choice)                # 5
    if got is None:
        return None
    d, shards, quant = got
    pack = pack_step(a, d, shards, llama, quant)       # 6
    vision = vision_step(a, pc, meta, llama, d, quant)  # 7
    cfg_path = write_config(a, pc, meta, pack, quant, ctx, models, vision, choice)   # 8
    # the tuning: asked for (--calibrate), or offered when nothing was tuned for this PC and model yet and someone
    # answers (--yes and --no-start setups are not held up by it)
    tuned = None
    cfg = read_json(cfg_path)
    offer = (cfg.get("backend") != "hip" and saved_calibration(cfg, any_context=True) is None and not a.yes and
             not a.no_start)
    if a.calibrate or (offer and ask(
            "Tune Maya for this PC now? It measures the split of the work between the CPU and the PCIe link and the "
            "CPU threads (about 10-15 minutes; later: ./abyss.sh --calibrate)", ["y", "n"], "y", a.yes) == "y"):
        if S.UI is not None:                           # (the screen lists the tuning as a step of its own)
            S.UI.step(9, "tuning Maya for this PC")
        tuned = calibrate_config(cfg_path)
    if tuned is False:
        say()
        warn("this PC is NOT tuned (the reason is above): ./abyss.sh --calibrate tries again")
    return cfg_path


def set_up_and_serve(a, prev: dict) -> tuple | None:
    """On the setup's screen: the setup, then Maya running on the same screen - its server's output there, Ctrl+C
    stops it.  (The config, the server's exit code or None when it was not started), or None when the model was not
    downloaded."""
    cfg_path = set_up(a, prev)
    if cfg_path is None or a.no_start:
        return cfg_path and (cfg_path, None)
    return cfg_path, serve_on_screen(cfg_path, a)


# ------------------------------------------------------------------------------------------------ main
def maya_version() -> str:
    return (HERE / "VERSION").read_text(encoding="utf-8").strip() if (HERE / "VERSION").exists() else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cuda", "hip"], help="GPU backend (HIP: experimental gfx1100/gfx1201/gfx1151 on Linux and Windows)")
    ap.add_argument("--setup", action="store_true", help="set up again instead of starting the installed model")
    ap.add_argument("--check", action="store_true",
                    help="with a model installed: a stability check of it (Abyss stopped, ~15-30 minutes) - long "
                         "prompts, a long conversation, Japanese, code, a tool call, the drafter - that writes "
                         "abyss-check.txt; before the setup (or with --setup / --gpu / --gpus): only check this PC")
    ap.add_argument("--quick", action="store_true", help="with --check: the short check (~5 minutes)")
    ap.add_argument("--finance", action="store_true", help="with --check: the Japanese finance rules check "
                    "(tools/finance_check.py) instead of the stability check")
    ap.add_argument("--long", action="store_true", help="with --check: also the biggest prompts the context allows")
    ap.add_argument("--no-start", action="store_true", help="set up, but do not start the dashboard")
    ap.add_argument("--yes", action="store_true",
                    help="take the recommended answers (the model download still needs --download-model)")
    ap.add_argument("--plain", action="store_true",
                    help="the setup in plain text, the answers typed (not on the setup's own screen); with "
                         "--terminal: no Textual, plain text redrawn every second")
    ap.add_argument("--download-model", action="store_true",
                    help="download the model without asking (the commands and the size are still printed)")
    ap.add_argument("--model", choices=list(MODELS), help=f"which download (default: asked, {next(iter(MODELS))} "
                                                          "recommended)")
    ap.add_argument("--no-vision", action="store_true", help="text only: no image encoder (saves its build and "
                                                             "its 1.1 GB of files)")
    ap.add_argument("--gguf-dir", help="use GLM-5.3-Flash GGUF files you already have: the folder with all of them, "
                                       "or the .gguf file (the first one of a split model) when the folder holds "
                                       "several models; it must be writable - the pack is written inside it "
                                       "(without this option the setup lists the ones it finds in ~/models*)")
    ap.add_argument("--models-dir", help="where downloaded models go, each in a folder named after it, e.g. "
                                         "<DIR>/Maya-L/ (default: Maya-data/models next to this folder); use "
                                         "a fast NVMe SSD with ~100 GB free")
    ap.add_argument("--gpu", type=int, help="run on this one GPU (CUDA: nvidia-smi; HIP: KFD topology order, on Windows hipInfo's)")
    ap.add_argument("--gpus", help=f"split the model's layers across these GPUs (up to {MAX_GPUS}), e.g. 0,1 or "
                                   "0,1,2,3")
    ap.add_argument("--context", type=int, help=f"context length in tokens (default {DEFAULT_CONTEXT})")
    ap.add_argument("--port", type=int, help="the dashboard's and the API's port (default 8080)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this machine only (default), 0.0.0.0 = also "
                                   "other devices on your network (set --api-key too)")
    ap.add_argument("--api-key", help="require this key from API clients and the dashboard")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="an engine setting kept in the config, e.g. STRATA_GLM_RAM_HEADROOM_GB=4 (README-MAYA.md)")
    ap.add_argument("--nvcc", help="the CUDA toolkit's nvcc to build with (default: the newest that fits your GPUs)")
    ap.add_argument("--host-compiler", help="the C++ compiler nvcc uses, e.g. g++-12 (when the default g++ is newer "
                                            "than your CUDA accepts)")
    ap.add_argument("--llama-dir", help="a llama.cpp checkout at the pinned commit, instead of downloading its source")
    ap.add_argument("--rebuild", action="store_true", help="compile the engine again")
    ap.add_argument("--repack", action="store_true", help="build the pack again")
    ap.add_argument("--report", action="store_true", help="write abyss-report.txt - this PC, the setup and the engine's "
                                                          "speed lines - to attach when you report a problem or a "
                                                          "speed (nothing is sent anywhere)")
    ap.add_argument("--bench", action="store_true", help="a standard speed test of the installed model (a few "
                                                         "minutes, with Abyss stopped): decode on three questions, "
                                                         "prefill at 2k / 8k tokens, where a token's time goes (the "
                                                         "likely bottleneck) and a quality check; writes "
                                                         "abyss-bench.txt to paste into an issue")
    ap.add_argument("--deep", action="store_true", help="with --bench: find the bottleneck - the GPU's timeline and a "
                                                        "sweep of the PCIe / CPU-lane share (a few minutes more)")
    ap.add_argument("--full", action="store_true", help="with --bench: three runs of each question and a 32k prompt")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's CPU lane for this PC (the PCIe share and the CPU threads, ~10-15 minutes), "
                         "then start the model (with --no-start: only tune)")
    ap.add_argument("--gpu-tune", action="store_true",
                    help="measure the engine's per-GPU kernel choices on this PC's GPU(s) (no model, about a minute "
                         "each), write the faster ones into the installed configs and abyss-gpu-tune.txt (a report to "
                         "paste into an issue)")
    ap.add_argument("--terminal", action="store_true",
                    help="Abyss Terminal: watch the server (speed, where a token's time goes, GPUs, memory, requests), "
                         "start / stop / switch models; closing it keeps the server running (docs/MANUAL.md)")
    ap.add_argument("--url", help="with --terminal or --connect: a server on another PC, e.g. http://192.168.1.20:8082")
    ap.add_argument("--key-env", metavar="NAME",
                    help="with --terminal / --connect --url: the environment variable that holds the server's API key")
    ap.add_argument("--interval", type=float, default=1.0, help="with --terminal: seconds between readings (0.5-10)")
    ap.add_argument("--layout", choices=("speed", "devices"),
                    help="with --terminal: the Overview - speed (big numbers) or devices (every GPU / SSD / CPU)")
    ap.add_argument("--once", action="store_true", help="with --terminal: print one reading and exit")
    ap.add_argument("--json", action="store_true", help="with --terminal: print one reading as JSON and exit")
    ap.add_argument("--demo", metavar="SCENARIO", help="with --terminal: recorded readings, no server (r9700, lowspec, "
                                                       "split, multi4, multi8, multi10, loading, starting, down)")
    ap.add_argument("--connect", choices=("claude", "opencode"),
                    help="print the settings that point Claude Code / OpenCode at the server that runs now; with "
                         "--write: write them (asked first, a backup kept), with --check: try one prompt and one tool "
                         "call the way the tool sends them (docs/MANUAL.md)")
    ap.add_argument("--write", action="store_true", help="with --connect: write the settings")
    ap.add_argument("--config", type=Path, help="the installed JSON config to start (its run-maya-<model>.sh does), or "
                                              "for --bench or --report (default: most recently used)")
    a = ap.parse_args()
    if a.config is not None:
        if not a.config.is_file():
            ap.error(f"config not found: {a.config}")
        if a.backend and read_json(a.config).get("backend", "cuda") != a.backend:
            ap.error(f"{a.config}: config backend does not match --backend {a.backend}")
    if not WIN and sys.prefix == sys.base_prefix and not os.environ.get("MAYA_SH"):
        # started as `python3 abyss.py`: a system Python takes no pip installs (PEP 668, "externally-managed-
        # environment") - abyss.sh makes the private .venv and runs this file with its Python
        say("Starting through ./abyss.sh, which runs Maya with its own Python environment (.venv) ...")
        sys.stdout.flush()
        os.execve("/bin/sh", ["/bin/sh", str(HERE / "abyss.sh"), *sys.argv[1:]], dict(os.environ, MAYA_SH="1"))
    if a.terminal:
        return terminal(a)
    if a.connect:
        import abyss_connect
        argv = [a.connect] + [f for f, on in (("--write", a.write), ("--check", a.check), ("--yes", a.yes)) if on]
        for flag, val in (("--url", a.url), ("--key-env", a.key_env), ("--config", a.config)):
            if val is not None:
                argv += [flag, str(val)]
        return abyss_connect.main(argv)
    version = maya_version()
    say(f"Abyss v{version} - big MoE models on your own GPU, RAM and SSD.")
    if a.report:
        return report(version, a.backend, a.config)
    if a.gpu_tune:
        return gpu_tune(a)
    pc_only = a.setup or a.gguf_dir or a.model or a.rebuild or a.repack or a.download_model or a.gpu is not None \
        or a.gpus
    if a.check and not pc_only:
        have = [a.config] if a.config is not None else all_configs()
        if a.backend:
            have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
        if have:
            if a.finance:
                return finance_check(have[0], version)
            return model_check(have[0], version, "quick" if a.quick else "long" if a.long else "default")
    if a.bench:
        have = [a.config] if a.config is not None else all_configs()   # (setup-deepseek.sh's abyss-*.json too)
        if a.backend:
            have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
        if not have:
            fail("Abyss is not set up here yet", f"run {ME} first")
        return bench(have[0], version, deep=a.deep, full=a.full)

    have = configs() if a.config is None else [a.config.resolve()]   # (--config: that one, no question)
    if a.backend:
        have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
    else:
        a.backend = "hip" if have and read_json(have[0]).get("backend") == "hip" else "cuda"
    setting_up = a.setup or a.check or a.gguf_dir or a.model or a.rebuild or a.repack or a.download_model
    if have and not setting_up and (a.calibrate or not a.no_start):
        pick = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {c.name}")
            pick = have[int(ask("Which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        if a.calibrate:
            refresh_engine(read_json(pick))            # (the engine the tuning measures is the current one)
            if not calibrate_config(pick):
                say()
                warn("this PC is NOT tuned (the reason is above): the model " +
                     ("keeps" if a.no_start else "starts with") + " the engine's own settings")
            if a.no_start:
                return 0
        return start(pick, a)
    if not have and a.calibrate:
        say("Nothing is set up yet: the setup runs first, then the tuning.")
    prev = read_json(have[0]) if have else {}
    if a.check:
        pc = check_pc(a)
        say()
        if a.backend == "hip":
            pick = (f"--gpus {','.join(str(g['index']) for g in pc['gpus'])}" if len(pc["gpus"]) == 2
                    else f"--gpu {pc['gpus'][0]['index']}")
            say(f"HIP prerequisites found. Run {ME} --backend hip {pick} to try the experimental port.")
        else:
            say(f"This PC can run Maya. Run {ME} without --check to set it up.")
        return 0
    rc = None                                          # the server's exit code, when it ran on the setup's screen
    if setup_screen(a):
        import setup_tui
        cfg_path, rc = setup_tui.run(lambda: set_up_and_serve(a, prev), version, ROOT / "maya-setup.log",
                                     ME) or (None, None)
    else:
        cfg_path = set_up(a, prev)
    if cfg_path is None:                               # the model was not downloaded (what to do next is above)
        return 0
    if rc is not None:
        return after_server(rc, a)
    if a.no_start:
        say()
        say(f"All set. Start it with {ME} (or " + (f"run-{cfg_path.stem}.bat" if WIN else f"./run-{cfg_path.stem}.sh")
            + ").")
        return 0
    return start(cfg_path, a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
