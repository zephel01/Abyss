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
    ./setup-deepseek.sh --model qwen35       Qwen3.5-122B-A10B, zephel01's IQ3_XXS-B2 for a 16 GB GPU + 64 GB RAM
    ./setup-deepseek.sh --model qwen38       Qwen3.8-Flash-Next, zephel01's GGUFs: --quant 12G32G (12 GB GPU + 32 GB RAM),
                                             16G64G (the default) or 24G64G
    ./setup-deepseek.sh --model glm          GLM-5.3-Flash, zephel01's GGUFs: --quant 32G96G (the default) or 32G128G
    ./setup-deepseek.sh --tools-only         only (re)create the Abyss-tools folder
    ./setup-deepseek.sh --dspark             also DeepSeek's DSpark drafter (6.9 GB): faster answers (--no-dspark: off)
    ./setup-deepseek.sh --model qwen38 --dflash   also the DFlash 2 drafter (0.65 GB): faster code (--no-dflash: off)

What it does:

  1. checks the PC: an NVIDIA GPU, nvcc, RAM
  2. Python packages into .venv; llama.cpp's source at the pinned commit (third_party/llama.cpp)
  3. compiles the engine (build/strata) for the GPU - arch 120a on RTX 50, whose MXFP4 kernels need it
  4. the model: unsloth's DeepSeek-V4-Flash-0731-GGUF quant into <data>/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<quant>
     (--model v41: zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF, 132G or 139G)
     (asks first, or --download-model), every file checked against Hugging Face's SHA-256
  5. the pack next to the shards (tools/iq_pack.py --compat-bf16)
  6. the run config abyss-<name>.json and the start script run-abyss-<name>.sh in the Abyss folder
  7. the DSpark drafter (V4 only; asked when the config is new, or --dspark): bleysg's GGUF of DeepSeek's block drafter into
     <data>/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF, its path in the config's env (STRATA_DS4_DSPARK)
  8. the Abyss-tools folder (abyss.sh start / stop / status / update / log / models) from ops/abyss-tools/

The data folder defaults to Abyss-data beside Abyss, the tools folder to Abyss-tools beside it.  The helpers (the PC
checks, pip, llama.cpp's source, resumable downloads) are setup.py's, inherited from Maya and Strata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIN = os.name == "nt"
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

REPO = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
# the models this setup installs: the repo, its default quant (a folder of the repo), the quants it offers when the repo
# is a fixed set, the config's model_name, the config name's prefix, the env the model's measurements were made with,
# the experts' size in GB at the default quant (for the RAM line)
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
    exe = ROOT / "build" / "strata"
    cmake = venv_bin("cmake")
    ninja = venv_bin("ninja")
    cc = cuda_archs(arch, gpus)
    nvcc = nvcc_arg or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    conf = [cmake, "-S", ROOT, "-B", ROOT / "build", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cc}",
            "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_COMPILER={nvcc}",
            f"-DSTRATA_GGML_DIR={ROOT / 'third_party' / 'llama.cpp'}"]
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
    S.run([cmake, "--build", ROOT / "build", "--target", "strata", "abyss_gpu_tune", "-j", str(max(2, (os.cpu_count() or 4)))],
          quiet=True)
    S.ok(f"engine: {exe}")
    return exe


def hip_engine(a):
    """AMD (--backend hip): the GPU check, ROCm and the engine (build-hip/strata) by abyss.py's HIP helpers (Maya's
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


def write_config(name: str, pack: Path, ctx: int, port: int, rewrite: bool, gpus=None,
                 extra: dict | None = None, model: dict | None = None) -> tuple[Path, Path]:
    model = model or MODELS["v4"]
    cfg = ROOT / f"abyss-{name}.json"
    run = ROOT / f"run-abyss-{name}.sh"
    if cfg.exists() and not rewrite:
        S.ok(f"{cfg.name} kept (--rewrite-config writes it again)")
        # ... with the libraries of the engine just checked or compiled: after a switch to another ROCm (setup again
        # with ROCM_PATH) the old one's lib first in LD_LIBRARY_PATH would load its runtime into the new engine - ROCm 7
        # and 10 share the sonames (libamdhip64.so.7, libhsa-runtime64.so.1) and the engine has no RUNPATH
        kept = json.loads(cfg.read_text(encoding="utf-8"))
        if (extra or {}).get("lib_dirs") and kept.get("lib_dirs") != extra["lib_dirs"]:
            kept["lib_dirs"] = extra["lib_dirs"]
            cfg.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
            S.ok(f"{cfg.name}: lib_dirs {', '.join(extra['lib_dirs'])} (the libraries the engine was compiled with)")
    else:
        cfg.write_text(json.dumps({
            "exe": str(ROOT / "build" / "strata"),
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
        }, indent=1) + "\n", encoding="utf-8")
        S.ok(f"{cfg.name} written (context {ctx}, port {port})")
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up DeepSeek V4 Flash / V4.1 Flash 1017 on Abyss")
    ap.add_argument("--model", choices=tuple(MODELS), default="v4", help="v4: DeepSeek V4 Flash 0731 (unsloth's GGUF, "
                    "the default); v41: DeepSeek V4.1 Flash 1017 (zephel01's LDLQ GGUF for a 32 GB GPU + 96 GB RAM, BETA); "
                    "ornith: Ornith-1.5-122B, ornith35: Ornith-1.5-35B-A3B (the official Q4_K_M, for an 8-16 GB GPU), qwen35: Qwen3.5-122B-A10B (zephel01's IQ3_XXS for a 16 GB GPU + 64 GB RAM); "
                    "qwen38: Qwen3.8-Flash-Next, glm: GLM-5.3-Flash (zephel01's GGUFs, --quant picks the PC size)")
    ap.add_argument("--quant", help="v4: a folder of " + REPO + " (default UD-Q2_K_XL, the tested one), or zephel01's 16G64G / 32G64G / 32G96G; v41: 132G "
                    "(default, speed) or 139G (quality); ornith: IQ3_XXS-B (default), 24G64G, 32G128G (Ornith-1.5-397B) or 16G64G (Ornith-1.5-397B on a 16 GB GPU); ornith35: Q4_K_M (default), Q5_K_M, Q6_K or Q8_0; qwen38: 12G32G, 16G64G (default) or 24G64G; glm: 32G96G (default) or 32G128G")
    ap.add_argument("--name", help="the model's name in abyss-<name>.json / run-abyss-<name>.sh (default dsv4-<quant>, "
                                   "dsv41-<quant>)")
    ap.add_argument("--data-dir", default=str(ROOT.parent / "Abyss-data"), help="where the models go (default: Abyss-data beside Abyss)")
    ap.add_argument("--tools-dir", default=str(ROOT.parent / "Abyss-tools"), help="the tools folder (default: Abyss-tools beside Abyss)")
    ap.add_argument("--context", type=int, default=65536, help="the context in tokens (default 65536)")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--backend", choices=("cuda", "hip"), help="cuda (NVIDIA) or hip (AMD: RX 7900 XT / XTX, RX 9070 / "
                    "AI PRO R9700, Ryzen AI Max's Radeon 8060S; ROCm 7). Default: cuda when an NVIDIA GPU is there, else hip")
    ap.add_argument("--gpu", type=int, help="the GPU, as nvidia-smi numbers it (default: the one with the most VRAM)")
    ap.add_argument("--gpus", help="several GPUs for a layer split, as nvidia-smi numbers them (\"0,1\"; the first is "
                                   "the main card), or \"all\": every NVIDIA GPU (default: one GPU, --gpu)")
    ap.add_argument("--arch", help="the CUDA architecture to compile for (default: the GPU's; 120a on RTX 50)")
    ap.add_argument("--nvcc", help="the CUDA compiler to build with (default: nvcc on PATH, else /usr/local/cuda); a V100 "
                                   "needs a CUDA 12 one, e.g. /usr/local/cuda-12.9/bin/nvcc")
    ap.add_argument("--download-model", action="store_true", help="download the model without asking")
    ap.add_argument("--yes", action="store_true", help="the defaults for every question (the download still needs --download-model)")
    ap.add_argument("--rewrite-config", action="store_true", help="write abyss-<name>.json again")
    ap.add_argument("--dspark", action="store_true", help="(V4) download DeepSeek's DSpark drafter (6.9 GB) and turn it on in "
                    "the config: drafts checked a step at a time, ~+28%% on swe-bench tasks on an AI PRO R9700 (~6.5 GB "
                    "of VRAM; one GPU, DeepSeek V4 Flash 0731). Asked when the config is new")
    ap.add_argument("--no-dspark", action="store_true", help="turn the DSpark drafter off in the config (the file stays)")
    ap.add_argument("--dflash", action="store_true", help="(qwen38) download the DFlash 2 drafter (0.65 GB) and turn it on "
                    "in the config instead of the MTP layer: faster code and lists, slower prose (docs/MANUAL.md, "
                    "Qwen3.8-Flash-Next); one GPU")
    ap.add_argument("--no-dflash", action="store_true", help="turn the DFlash 2 drafter off in the config (the file stays)")
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
    quant = a.quant or model["quant"]
    if model["quants"] and quant not in model["quants"]:
        ap.error(f"--model {a.model} has the quants {', '.join(model['quants'])}")
    own = quant in model.get("variants", {})   # (the quant is a PC size and picks zephel01's repo for it)
    if own:
        model = {**model, **model["variants"][quant]}
    if "prefix_tmpl" in model:
        model = {**model, "prefix_file": model["prefix_tmpl"].format(q=quant)}
    repo = model["repo"]
    if a.dspark and a.model != "v4":
        ap.error("--dspark: the DSpark drafter is set up for DeepSeek V4 only (V4.1: docs/deepseek41/README.md)")
    if (a.dflash or a.no_dflash) and a.model != "qwen38":
        ap.error("--dflash: the DFlash 2 drafter is Qwen3.8-Flash-Next's (--model qwen38)")
    name = a.name or model["prefix"] + "-" + quant.lower().removeprefix("ud-")
    data = Path(a.data_dir).expanduser().resolve()
    mdir = data / "models" / repo / quant

    S.step(1, "the PC")
    gpu = S.gpu_info(a.gpu) if a.backend != "hip" else None
    backend = a.backend or ("cuda" if gpu is not None else "hip" if S.amd_gpus() else None)
    if backend is None:
        S.fail("no NVIDIA or supported AMD GPU found", "Abyss runs on NVIDIA GPUs (CUDA), or AMD ones with --backend hip")
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
    if backend == "cuda" and len(gpus) == 1 and gpu.get("count", 1) > 1:
        S.say(f"  ({gpu['count']} NVIDIA GPUs here: --gpus all (or --gpus 0,1) splits the layers across them)")
    if own and a.model != "ornith" or quant in ("24G64G", "32G128G", "16G64G"):
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is made for a {model['vram']} GB GPU + {model['ram']} GB RAM: with "
             f"VRAM, ~{model['gb']} GB holds every expert; less reads from the SSD)")
        if S.ram_gb() + (gpus[0].get("vram_gb") or 0 if gpus else 0) < model["gb"] * 0.8:
            S.say(f"  (this PC has less: a smaller --quant ({', '.join(model['variants'])}) may suit it better)")
    elif a.model == "v4":
        S.ok(f"RAM {S.ram_gb():.0f} GB (with VRAM, ~100 GB holds every expert of the 97 GB quant; less reads from the SSD)")
    elif a.model == "ornith35":
        S.ok(f"RAM {S.ram_gb():.0f} GB (the {quant} file is ~{model['gb']} GB: VRAM + RAM hold it all on an 8-16 GB GPU + "
             "32 GB RAM PC)")
    elif a.model in ("ornith", "qwen35"):
        S.ok(f"RAM {S.ram_gb():.0f} GB (made for a 16 GB GPU + 64 GB RAM: with VRAM, ~60 GB holds every expert of the "
             f"{model['gb']} GB file; less reads from the SSD)")
    else:
        S.ok(f"RAM {S.ram_gb():.0f} GB (made for a 32 GB GPU + 96 GB RAM: the experts that do not fit there and the "
             "Engram table are read from the SSD - keep the model on a fast NVMe)")

    S.step(2, "Python packages and llama.cpp's source")
    S.pip_install(S.PY_PACKAGES, ", ".join(S.PY_PACKAGES))
    S.ok(f"llama.cpp: {S.get_llama_cpp()}")

    S.step(3, "the engine")
    if backend == "cuda":
        build_engine(a.arch, gpus, a.nvcc)

    S.step(4, f"the model: {repo} {quant}")
    files = quant_files(quant, repo, model.get("folder"), model.get("prefix_file", ""))
    total = sum(f[1] for f in files)
    S.say(f"  {len(files)} files, {total / 1e9:.1f} GB -> {mdir}")
    missing = [f for f in files if not (mdir / Path(f[0]).name).exists()]
    if missing:
        free = shutil.disk_usage(mdir.parent if mdir.parent.exists() else data.parent).free
        S.say(f"  to download: {sum(f[1] for f in missing) / 1e9:.1f} GB (free on that disk: {free / 1e9:.0f} GB)")
        if not a.download_model and S.ask("  Download it now?", ["y", "n"], "n", a.yes) != "y":
            S.fail("the model is not downloaded", "run again with --download-model, or put the files into " + str(mdir))
    for path, size, sha in files:
        dst = mdir / Path(path).name
        S.download(HF_FILE.format(repo=repo, path=path), dst, dst.name)
        if sha and not verified(dst, sha):
            S.fail(f"{dst.name} does not match Hugging Face's SHA-256", "delete it and run again")
    S.check_shards([mdir / Path(f[0]).name for f in files])
    S.ok("every file checked")

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
    if a.model in ("qwen38", "glm", "ornith35") and S.ram_gb() < 160:
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
    cfg, _ = write_config(name, pack, a.context, a.port, a.rewrite_config,
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
    else:
        S.ok("DSpark drafter: set up for DeepSeek V4 only" + (" (V4.1: by hand, docs/deepseek41/README.md)" if a.model == "v41"
             else " (Ornith drafts with its own NextN block from the shards)" if a.model in ("ornith", "ornith35")
             else " (Qwen3.8 drafts with its own MTP block from the shards)" if a.model == "qwen38" else ""))
    if a.model == "qwen38":
        dflash_step(a, data, cfg, gpus)

    S.step(8, "Abyss-tools")
    make_tools(tools)

    S.say()
    S.say(f"Done. Start it with:  {tools / 'abyss.sh'} start {name}")
    S.say(f"then open http://127.0.0.1:{a.port}/  (the manual: {ROOT / 'docs' / 'MANUAL.md'})")
    S.say(f"Once, with it stopped: {tools / 'abyss.sh'} --calibrate  tunes the CPU lane / PCIe split for this PC "
          "(~10-15 minutes; +12% on an RTX 4090, issue #39)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
