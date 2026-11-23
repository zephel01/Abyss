# Abyss on Windows 11 (NVIDIA and AMD)

[日本語](WINDOWS.ja.md)

How to install and run Abyss on Windows 11, with an NVIDIA GeForce or an AMD Radeon card. This page is written for
v1.3.189.

**Read this first.** We develop and test Abyss on Linux. There is no Windows PC here, so nothing on this page was run
on Windows by us. The steps come from the setup's code and from what users reported in GitHub issues; each report is
linked (#number). Where nobody has reported a step yet, the page says **not reported yet**.

## Contents

- [1. What you need](#1-what-you-need)
- [2. Before the setup (both GPUs)](#2-before-the-setup-both-gpus)
- [3. NVIDIA](#3-nvidia)
- [4. AMD Radeon](#4-amd-radeon)
- [5. Which model for your PC](#5-which-model-for-your-pc)
- [6. Start, measure, tune](#6-start-measure-tune)
- [7. RAM, the page file and other Windows settings](#7-ram-the-page-file-and-other-windows-settings)
- [8. Updating](#8-updating)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Reporting a result or a problem](#10-reporting-a-result-or-a-problem)
- [11. What has not been checked on Windows](#11-what-has-not-been-checked-on-windows)

## 1. What you need

| | NVIDIA | AMD |
| --- | --- | --- |
| Windows | Windows 11, 64-bit | Windows 11, 64-bit |
| GPU | RTX 30 / 40 / 50 for the prebuilt engine. Other cards (RTX 20, V100, ...) need their own build | RX 7900 XT / XTX (gfx1100), RX 9070 / 9070 XT / AI PRO R9700 (gfx1201), RX 9060 / 9060 XT (gfx1200), Ryzen AI Max's Radeon 8050S / 8060S (gfx1151). The siblings RX 7800 XT / 7700 XT (gfx1101), RX 7600 (gfx1102) and Radeon 890M (gfx1150) build, not validated. RDNA2 (RX 6000) and older: not supported |
| Driver | **580 or newer required**, 610 or newer recommended (prebuilt engine) | AMD Software: Adrenalin Edition (https://www.amd.com/en/support) |
| Visual Studio Build Tools | prebuilt engine: **none**. Own build: 2022 (or 2019) with "Desktop development with C++" - CUDA 13 does not accept Visual Studio 2026 | **needed**: 2022 or 2026 with "Desktop development with C++" (ROCm's compiler uses its headers and libraries) |
| CUDA / ROCm | prebuilt engine: **none** (CUDA 13.3 libraries come with it). Own build: CUDA Toolkit 12.8 or newer | AMD's ROCm SDK 7.14.1 (TheRock) - the setup installs it into Abyss's own `.venv` |
| Python | 64-bit, 3.10 or newer | the same |
| CPU | AVX2 | AVX2 (the AMD setup stops without it) |

Reported on Windows so far:

| Card | Reports |
| --- | --- |
| RTX 4090 24 GB + 96 GB RAM, own build (MSVC + CUDA) | DeepSeek V4 Flash 0731 (#31, #46), Qwen3.8-Flash-Next UD-Q4_K_XL (#39) |
| 2x RTX 5060 Ti 16 GB + 128 GB RAM, own build, CUDA 13.3 | GLM-5.3-Flash (#40) |
| RX 7900 XT 20 GB + 128 GB RAM, ROCm SDK 7.14.1 | DeepSeek V4 0731, V4.1 (#2-#6, #9, #12) |
| RX 9060 XT 16 GB + 64 GB RAM, ROCm SDK 7.14.1, VS 2026 Build Tools (MSVC 14.51), Adrenalin 26.11.2; v1.3.189 builds and runs (MMQ built optimized) | Ornith-1.5-122B, Qwen3.8-Flash-Next 12G32G / 16G64G, SenseNova-Loop-9B (#43, #47, #49) |

The prebuilt NVIDIA engine on Windows: **not reported yet**.

### Disk space

- The model: from 5.6 GB (the 9B models) to about 100 GB (the 16G64G-96G sizes); DeepSeek V4.1 is 207-283 GB. Each
  size is in the [README](../README.md#disk-space).
- About 3 GB more for the engine build, llama.cpp's source and the Python packages.
- AMD: the ROCm SDK in `.venv`. Its first `rocm-sdk init` unpacks about 3 GB.
- Put the model on an NVMe SSD. When the model does not fit in VRAM + RAM, the SSD's speed sets the decode speed.

### RAM and the page file

Windows counts the pinned RAM that holds the experts (the "RAM tier") against its **commit limit**: RAM + page file.
With a small page file, experts are read from the SSD even while RAM is free. Set the page file up before the first
run: [section 7](#7-ram-the-page-file-and-other-windows-settings).

## 2. Before the setup (both GPUs)

All commands are for **PowerShell** (Windows Terminal or "Windows PowerShell"). Run them as your own user; an
administrator window is not needed for the setup.

1. **Python** (64-bit, 3.10 or newer). The setup's `.bat` files print the same command when Python is missing.

   ```powershell
   winget install -e --id Python.Python.3.12 --scope user
   ```

   With the installer from https://www.python.org/downloads/ instead, tick "Add python.exe to PATH".
   Open a new PowerShell window after installing.

2. **Git.** Updates (the dashboard's About > Update, or `git pull`) need a git checkout; a ZIP download cannot update
   itself.

   ```powershell
   winget install -e --id Git.Git
   ```

3. **Get Abyss.** A short path without spaces is safest: the HIP build passes some compiler paths only when the path
   has no space. The reporters used `C:\AI\...`, `D:\AI\...` and `C:\Users\<name>\...`.

   ```powershell
   mkdir C:\AI -Force
   cd C:\AI
   git clone https://github.com/zephel01/Abyss.git
   cd C:\AI\Abyss
   ```

   The models go to `C:\AI\Abyss-data`, the tools folder to `C:\AI\Abyss-tools` (beside Abyss).
   `--data-dir D:\Abyss-data` puts the models on another drive.

4. **The page file**: [section 7](#7-ram-the-page-file-and-other-windows-settings). It needs a restart, so do it now.

The two setup files:

| File | What it sets up |
| --- | --- |
| `setup-deepseek.bat` | every model in the README's table (DeepSeek V4 / V4.1, Qwen3.8, Ornith, GLM-5.3-Flash with `--model glm`, the 9B models, ...). Runs `tools\abyss_setup.py`. **This page uses it.** |
| `START-ABYSS.bat` | starts an installed model (`--config`), and runs `--bench`, `--calibrate`, `--check`, `--report`, `--gpu-tune`. Without a config it is Maya's GLM-5.3-Flash installer |

Both make the private Python environment `.venv` in the Abyss folder the first time. Nothing is installed
system-wide. `.\setup-deepseek.bat --help` lists every option.

## 3. NVIDIA

### 3.1 The prebuilt engine (no CUDA toolkit, no Visual Studio)

From v1.3.164 the setup downloads an engine built by GitHub Actions when it finds no CUDA toolkit (`nvcc`) or no
Visual Studio C++ compiler. That is the easy way on Windows.

| | |
| --- | --- |
| File | `abyss-engine-cuda-windows-x64.zip` from the GitHub release of this version (else the latest), checked against the release's `SHA256SUMS`, unpacked into `build\` |
| GPUs | RTX 30 (sm_86), RTX 40 (sm_89), RTX 50 (sm_120a), in one engine. Other cards stop with the reason |
| Inside | `abyss.exe`, `abyss_gpu_tune.exe`, and NVIDIA's cuBLAS / cuBLASLt DLLs (CUDA 13.3) |
| Driver | 580 or newer required (the setup stops below it), 610 or newer recommended |
| Not yet | images: the image encoder still needs the CUDA toolkit, so with the prebuilt engine the models read text only |

1. Check the driver:

   ```powershell
   nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
   ```

   Below 580: update from https://www.nvidia.com/drivers and restart.

2. Run the setup with the model for your PC ([section 5](#5-which-model-for-your-pc)). For example, a 16 GB card + 64
   GB RAM:

   ```powershell
   cd C:\AI\Abyss
   .\setup-deepseek.bat --model qwen38 --download-model
   ```

   It checks the PC, installs the Python packages, downloads the prebuilt engine, downloads the model (checking
   every file against Hugging Face's SHA-256), builds the pack, writes the config `abyss-<name>.json` and measures the
   GPU's kernel choices (`--gpu-tune`, about a minute). Every step is skipped when it is already done, so you can run
   it again after a stop.

3. With a CUDA toolkit and Visual Studio installed, the setup compiles instead. To take the prebuilt engine anyway:

   ```powershell
   .\setup-deepseek.bat --model qwen38 --download-model --prebuilt
   ```

Then go to [section 6](#6-start-measure-tune).

### 3.2 Building the engine yourself

For a card the prebuilt engine does not cover, for images, or to build from your own checkout. Users have built it
this way on an RTX 4090 (#31, #39, #46) and RTX 5060 Ti (#40).

1. **Visual Studio 2022 Build Tools** with "Desktop development with C++". CUDA 13 accepts Visual Studio 2019 and 2022
   only; the setup does not pick a Visual Studio 2026 for a CUDA build.

   ```powershell
   winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

2. **The CUDA Toolkit** for Windows, 12.8 or newer (RTX 50 needs 12.8+; #40 built with 13.3):
   https://developer.nvidia.com/cuda-downloads. The setup finds `nvcc` on PATH, through `CUDA_PATH`, or in
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*`.

3. Run the setup as in 3.1, without `--prebuilt`. It compiles `build\abyss.exe` for your card. RTX 50 gets `120a`
   (not `120`: the MXFP4 kernels need it, #40).

   If the build cannot find the C++ compiler, run the setup from the "x64 Native Tools Command Prompt for VS 2022"
   (Start menu) instead of PowerShell: `cd /d C:\AI\Abyss` and then `setup-deepseek.bat ...`. (We could not check
   whether this is needed.)

Older versions needed local patches to build on Windows. They are all upstream now (#31: v1.3.120, #39: v1.3.130,
#46: "zero local patches" on v1.3.147), and CI builds the Windows CUDA engine for every release.

## 4. AMD Radeon

The HIP engine is experimental and reads text only. On Windows it runs on one discrete card (or the Ryzen AI Max's
iGPU alone). The Linux "iGPU lane" (a discrete Radeon plus the APU's iGPU) does not exist on Windows.

### 4.1 Install

1. **The driver:** AMD Software: Adrenalin Edition (https://www.amd.com/en/support). Restart.
2. **Visual Studio Build Tools** with "Desktop development with C++". The command the setup prints:

   ```powershell
   winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

   Visual Studio 2026 (MSVC 14.51) works too from v1.3.131 (#43: VS 2026 Build Tools on an RX 9060 XT). When both are
   installed, the setup takes 2022.
3. **Python and Git:** [section 2](#2-before-the-setup-both-gpus).
4. **Run the setup** with `--backend hip` and the model for your PC:

   ```powershell
   cd C:\AI\Abyss
   .\setup-deepseek.bat --backend hip --model ornith --download-model
   ```

### 4.2 What the AMD setup does by itself

- **Installs AMD's ROCm SDK into `.venv`.** It asks first, then runs pip from repo.amd.com: ROCm SDK 7.14.1
  (TheRock's wheels) with the kernels for your card's architecture. Nothing is installed system-wide.
- **`rocm-sdk init`.** The first run unpacks the compiler, headers and libraries (about 3 GB, a few minutes):
  `AMD's ROCm SDK: rocm-sdk init (the first time it unpacks about 3 GB) ...`. Later runs only check.
- **Fixes one SDK header for Visual Studio 2026.** With MSVC 14.50 or newer, ROCm 7.14.1's Clang 23 stops at
  `'isgreater' cannot overload` (#2, #43). The setup moves one `#include` line in the SDK's
  `__clang_hip_runtime_wrapper.h` inside `.venv` (LLVM's own fix, llvm-project 01d3932, not in a Windows SDK yet), once,
  and keeps the original as `__clang_hip_runtime_wrapper.h.abyss-orig`. `ABYSS_NO_ROCM_HEADER_FIX=1` leaves the SDK as
  it is. From v1.3.131.
- **Builds `build-hip\abyss.exe`** with Ninja and ROCm's clang, for gfx1100, gfx1201, gfx1151 and your card's
  architecture (#43's build: `gfx1100;gfx1201;gfx1151;gfx1200`). It takes a while the first time. The build prints
  many warnings (#49); they are harmless. It copies ROCm's `amdhip64*.dll` next to the engine, so the engine does not
  load the driver's copy from System32.
- **Builds with `-O3`.** Some Windows toolchains (ROCm SDK 7.14.1 + CMake + Ninja) leave CMake's Release HIP flags
  empty. Before v1.3.29 every kernel was then built at `-O0` (1.3 instead of ~46 tok/s, #3); before v1.3.188 the MMQ
  prompt kernels still were (prompts ~100x+ slower, #49). From v1.3.188 every HIP kernel gets `-O3 -DNDEBUG`.
- **Writes the config** with `"backend": "hip"` and these `env` values: `STRATA_GLM_RAM_HEADROOM_GB` 16,
  `GPU_MAX_HW_QUEUES` 2, `STRATA_GLM_RESERVE_MB` 3072 (a card that drives a display), `STRATA_GLM_SPLIT` 0 (one card).
  It prints the Windows tuning of #4 ([section 7](#7-ram-the-page-file-and-other-windows-settings)).
- **Turns hipBLASLt off.** ROCm SDK 7.14.1's hipBLASLt crashed inside `amdhip64_7.dll` (#49). The engine does its own
  prompt GEMMs since v1.3.182, and from v1.3.188 the server and `--bench` start the engine with
  `ROCBLAS_USE_HIPBLASLT=0` on Windows. A value in the shell or the config's `"env"` wins.
- **Runs `--gpu-tune`** (about a minute) and writes `abyss-gpu-tune.txt`.

### 4.3 The ROCm SDK by hand

The setup installs the SDK only when it finds **no** ROCm. It stops instead when it finds an older one:

- **AMD's HIP SDK for Windows (ROCm 7.x) installed system-wide** (`C:\Program Files\AMD\ROCm\...`, `HIP_PATH`): the
  setup stops with `requires HIP 7.14 or newer` (#43: a ROCm 7.2 install).
- **`ROCM_PATH` set** to an older ROCm: the same stop.

Fix: install the SDK into Abyss's `.venv` by hand. The setup takes `.venv`'s SDK before `HIP_PATH` and
`C:\Program Files\AMD\ROCm`; only `ROCM_PATH` comes before it. Check, then install (the command from #43):

```powershell
cd C:\AI\Abyss
$env:ROCM_PATH; $env:HIP_PATH        # what is set now (empty lines: nothing)
Remove-Item Env:ROCM_PATH -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel,device-gfx1200]==7.14.1"
.\setup-deepseek.bat --backend hip --model ornith --download-model
```

`.venv` exists after the first run of `setup-deepseek.bat`. `Remove-Item` clears `ROCM_PATH` for this window only;
to remove it for good, delete it in "Environment Variables" (`rundll32 sysdm.cpl,EditEnvironmentVariables`).
Uninstalling the system-wide HIP SDK is not needed.

Replace `device-gfx1200` with your card's architecture. Several cards: one entry each, separated by commas
(`device-gfx1100,device-gfx1201`).

| Card | Architecture | pip extra |
| --- | --- | --- |
| RX 7900 XT / XTX | gfx1100 | `device-gfx1100` |
| RX 7800 XT / 7700 XT | gfx1101 | `device-gfx1101` |
| RX 7600 | gfx1102 | `device-gfx1102` |
| RX 9070 / 9070 XT, AI PRO R9700 | gfx1201 | `device-gfx1201` |
| RX 9060 / 9060 XT | gfx1200 | `device-gfx1200` |
| Radeon 8050S / 8060S (Ryzen AI Max) | gfx1151 | `device-gfx1151` |
| Radeon 890M | gfx1150 | `device-gfx1150` |

(#2 used the older form `rocm[libraries,devel,gfx1100]`; the setup uses `device-<arch>`.)

### 4.4 Check that the build is optimized

After the first run, the engine's log and `--bench` say `MMQ kernels built optimized`. `built WITHOUT optimization
(-O0)` means an engine from before v1.3.188: update and run the setup again ([section 8](#8-updating)). By hand
(from docs/amd):

```powershell
cd C:\AI\Abyss
Select-String -Path build-hip\CMakeCache.txt -Pattern "^CMAKE_HIP_FLAGS"
Select-String -Path build-hip\build.ninja -Pattern "mmq-instance-q4_k" -Context 0,8 | Select-Object -First 1
```

The `FLAGS =` line of the second must contain `-O3`.

### 4.5 What is different from Linux

- **No iGPU lane.** The CPU computes the RAM-tier experts, also with a Ryzen AI Max beside a discrete card.
- **GPU numbers** are those of ROCm's `hipInfo`. `"gpu_pci"` is not used. After adding or removing a card, run the
  setup again. A card can also be named by its UUID in `"gpu"` (`build-hip\abyss.exe --list-gpus` lists them);
  whether AMD's driver on Windows reports one is not checked yet - a card without one is listed with `-` and is named
  by its number.
- **`PCIe ?` in `--bench`.** The PCIe link cannot be read for an AMD card on Windows (#43, #47, #49). The rest of the
  benchmark works.
- **`HSA_OVERRIDE_GFX_VERSION`** is a Linux setting; it does nothing on Windows.
- **Ryzen AI Max alone:** Windows gives the iGPU a fixed share of the memory. The setup warns when that share is smaller
  than the RAM Windows keeps, and asks you to raise "Variable Graphics Memory" (AMD Software: Performance > Tuning, or
  the BIOS), e.g. 96 GB on a 128 GB PC. **Not reported yet** on Windows.

More on AMD (Linux included): [docs/amd](amd/README.md).

## 5. Which model for your PC

The setup picks the size for your PC when you leave out `--quant`: it shows `this PC: 16 GB GPU + 64 GB RAM ->
--quant 16G64G (the others: ...)` and asks. The models, one line each (AMD: add `--backend hip`):

| GPU + RAM | Model | Command | Config it writes |
| --- | --- | --- | --- |
| 12 GB + 32 GB | Qwen3.8-Flash-Next 12G32G | `.\setup-deepseek.bat --model qwen38 --quant 12G32G --download-model` | `abyss-qwen38-12g32g.json` |
| 16 GB + 64 GB | Qwen3.8-Flash-Next 16G64G | `.\setup-deepseek.bat --model qwen38 --download-model` | `abyss-qwen38-16g64g.json` |
| 16 GB + 64 GB | Ornith-1.5-122B IQ3_XXS-B | `.\setup-deepseek.bat --model ornith --download-model` | `abyss-ornith-iq3_xxs-b.json` |
| 8-16 GB + 32 GB | Ornith-1.5-35B-A3B Q4_K_M | `.\setup-deepseek.bat --model ornith35 --download-model` | `abyss-ornith35-q4_k_m.json` |
| 10-12 GB, all in VRAM | SenseNova-Loop-9B Q4_K_M | `.\setup-deepseek.bat --model sensenova-loop-9b --download-model` | `abyss-sensenova-loop-9b-q4_k_m.json` |
| 24 GB + 64 GB | Qwen3.8-Flash-Next 24G64G | `.\setup-deepseek.bat --model qwen38 --quant 24G64G --download-model` | `abyss-qwen38-24g64g.json` |
| 32 GB + 96 GB | GLM-5.3-Flash 32G96G | `.\setup-deepseek.bat --model glm --download-model` | `abyss-glm-32g96g.json` |
| about 100 GB VRAM + RAM | DeepSeek V4 Flash 0731 UD-Q2_K_XL (the default without `--model`) | `.\setup-deepseek.bat --download-model` | `abyss-dsv4-q2_k_xl.json` |

- The last line of the setup names the config (`Done. Start it with: ...\abyss.sh start <name>`: the config is
  `abyss-<name>.json` in the Abyss folder). Use that name if it differs from the table.
- Every size and model, with speed and quality: the [README](../README.md) ("Pick a model for your PC") and
  [docs/MODELS](MODELS.md). 8-12 GB cards: [docs/SMALL-VRAM](SMALL-VRAM.md).
- Numbers users reported on Windows, for reference (`--bench` decode): RX 9060 XT 16 GB + 64 GB, Ornith-122B 16-17
  tok/s (#43), Qwen3.8 12G32G 28 tok/s (#47), Qwen3.8 16G64G 22 tok/s (#49; 23.1 on v1.3.189, where 2K prompts went 9 -> 391 tok/s), SenseNova-Loop-9B 31.7
  tok/s with 8K prompts at 1,014 tok/s (#49, v1.3.189, Adrenalin 26.11.2); RTX 4090 + 96 GB, DeepSeek V4 0731 37-39
  tok/s without the DSpark drafter (#46).

## 6. Start, measure, tune

### Start

```powershell
cd C:\AI\Abyss
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json
```

Then open http://127.0.0.1:8082/ (the dashboard and the OpenAI / Anthropic-style API). Stop it with Ctrl+C in that
window.

- **Give `--config`.** The setup's last lines say `...\Abyss-tools\abyss.sh start <name>`; `abyss.sh` is a bash
  script and does not run in PowerShell. Use `START-ABYSS.bat --config abyss-<name>.json` (#43). Since v1.3.139,
  `START-ABYSS.bat` without `--config` also offers the configs `setup-deepseek.bat` wrote (before, it started Maya's
  GLM setup).
- The config is `abyss-<name>.json` in the Abyss folder; the engine's log is `abyss-<name>.log` beside it.
  What each key does: [docs/SETTINGS](SETTINGS.md).

### Measure and tune (with Abyss stopped)

```powershell
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --bench          # a few minutes: speed, the bottleneck, a quality check
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --bench --deep   # + a PCIe / CPU-lane share sweep
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --calibrate --no-start   # tunes the CPU lane / PCIe split, ~10-15 minutes
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --check          # a stability check, ~15-30 minutes
.\START-ABYSS.bat --gpu-tune                                           # the per-GPU kernel choices, a minute, no model
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --report         # writes abyss-report.txt
```

- `--bench` writes `abyss-bench.txt`, `--report` `abyss-report.txt`, `--gpu-tune` `abyss-gpu-tune.txt`, all in the
  Abyss folder. Nothing is sent anywhere. What the bench shows: [docs/BENCH](BENCH.md).
- `--calibrate` gave +12% on an RTX 4090 (#39: 50.2 -> 56.5 tok/s). Run it once after the setup, and again after a
  change to the PC.
- For coding agents, temperature 0 raised the drafter's acceptance on the 4090 (#39). See the manual's coding-tools
  section ([MANUAL](MANUAL.md#coding-tools-claude-code-and-opencode---connect)).

### A setting for one run (PowerShell)

`$env:NAME="value"` sets a variable for the rest of this PowerShell window; the engine started from it sees it.
`Remove-Item` clears it again (the form #49 used):

```powershell
$env:STRATA_GLM_TIMING="1"; $env:STRATA_GLM_PREFILL_PROF="1"; .\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --bench
Remove-Item Env:STRATA_GLM_TIMING, Env:STRATA_GLM_PREFILL_PROF -ErrorAction SilentlyContinue
```

To keep a setting, put it in the config's `"env"` instead (open it with `notepad .\abyss-qwen38-16g64g.json`; values
are strings in quotes, for example `"STRATA_GLM_RAM_HEADROOM_GB": "6"`) and start Abyss again.

## 7. RAM, the page file and other Windows settings

### The commit limit (RAM + page file)

Windows charges pinned RAM, and under WDDM every GPU allocation, to the commit limit. When that limit is the smaller
one, the engine's log says, for example:

```
glm fast: WARNING - Windows' commit limit (RAM + page file) caps the RAM tier 9 GB below the free RAM (...): those
experts are read from the SSD. Raise the page file's maximum size by at least 13 GB (System > Advanced system
settings > Performance > Virtual memory) and restart Abyss
```

(the line as of v1.3.174; #43 saw it with a RAM tier stuck at 22.8 GB on a 64 GB PC.)

1. See the commit limit (`TotalVirtualMemorySize`, in KB) and the page files:

   ```powershell
   Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, TotalVirtualMemorySize, FreeVirtualMemory
   Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage
   ```

2. Open the page file settings:

   ```powershell
   SystemPropertiesPerformance.exe
   ```

   Advanced > Virtual memory > Change. Untick "Automatically manage paging file size for all drives", pick the drive,
   choose "Custom size", set the sizes in MB, press Set, then OK, and restart Windows.
3. How large: at least what the warning asks for. The reports used **64 GB** (65536 MB): #4 a fixed 64 GB with 128 GB
   RAM; #43 24 GB initial / 64 GB maximum with 64 GB RAM - the RAM tier went from 22.8 to 33.9 GB. The drive needs that
   much free space.
4. From v1.3.174 the engine also counts the room a page file can still grow (system managed, or initial < maximum);
   `STRATA_PAGEFILE_GROWTH=0` counts only its current size.
5. Then, if the log still shows the RAM tier short of what the RAM allows, lower the headroom in the config's `"env"`:
   #43 used `"STRATA_GLM_RAM_HEADROOM_GB": "6"`, #4 used `"4"` (the AMD setup writes 16). Keep some room for Windows
   and your other programs.

The RX 7900 XT + 128 GB tuning that kept every expert of DeepSeek V4 in VRAM + RAM (#4), which the AMD setup prints:
a 64 GB page file, `STRATA_GLM_RAM_HEADROOM_GB=4`, `STRATA_GLM_RESERVE_MB=512` (only when no display is attached to
that card) and `--prefill 8192`.

### NVIDIA: turn off the Sysmem Fallback

In the NVIDIA Control Panel, Manage 3D settings, set "CUDA - Sysmem Fallback Policy" to "Prefer No Sysmem Fallback".
With the default, the driver moves VRAM allocations to host RAM without an error once the card is full (a browser or
the desktop took some), and decode slows down a lot. Abyss on Windows keeps 1.5-2 GB of VRAM free (v1.3.195+) for the same reason.

### Pinned memory that runs out before the RAM

The GPU driver can refuse more pinned memory before the RAM is full:

- NVIDIA (#31, RTX 4090, `STRATA_GLM_RAM_GB=72`): prompts from ~20K tokens failed and ~40K crashed. Since v1.3.107 the
  engine moves the prompt's CPU share to PCIe and says `the CPU lane's pinned host buffers did not allocate`. On that
  PC `"STRATA_GLM_PREFILL_CPU": "0"` was also slightly faster (#31, #46).
- AMD (#49, RX 9060 XT): the start stopped at `pinned disk staging did not allocate`. Since v1.3.181 the RAM tier gives
  slots back instead (`gave back ... GB of RAM-tier slots`).

### Intel CPUs with P-cores and E-cores

`STRATA_GLM_CPU_PIN=0-15` in the config's `"env"` puts the CPU lane's workers on the P-cores' threads (an i7-14700K
has 16 P-core threads, 0-15; #4, #12). From v1.3.56 those workers also run without Windows' power throttling
(EcoQoS). **Do not** limit the whole Abyss process to the P-cores (Task Manager's "Set affinity"): that starved the
GPU runtime's threads, 37 -> 19 tok/s and worse (#9, #12).

### The DSpark drafter on a 24 GB card

On an RTX 4090, DeepSeek V4 ran faster without the DSpark drafter (37-39 tok/s against 33-38 with it): its 6.5 GB of VRAM do
more as expert slots (#46). The engine turns DSpark on by itself only on 48 GB+ cards.

## 8. Updating

- The dashboard: About > Update. It needs a git checkout without local changes.
- By hand:

  ```powershell
  cd C:\AI\Abyss
  git pull
  .\setup-deepseek.bat --backend hip --model qwen38      # AMD; NVIDIA: the same without --backend hip
  ```

  The setup compiles only what changed (AMD) or downloads the new prebuilt engine (NVIDIA), and keeps the model, the
  pack and the config (`--rewrite-config` writes the config again, keeping the port, host, API key and MCP servers).
  A start (`START-ABYSS.bat --config ...`) and, since v1.3.139, `--bench` also rebuild a changed engine first.

## 9. Troubleshooting

| Symptom | Cause | Fix | Issue |
| --- | --- | --- | --- |
| `64-bit Python 3.10 or newer is needed` | no Python, or only the Microsoft Store stub | `winget install -e --id Python.Python.3.12 --scope user`, open a new window, run the `.bat` again | - |
| `.\setup-deepseek.bat` is "not recognized" in PowerShell | not in the Abyss folder, or no `.\` | `cd C:\AI\Abyss`, then `.\setup-deepseek.bat` | - |
| NVIDIA: the setup stops at the driver | driver older than 580 (the prebuilt engine is CUDA 13.3) | update from https://www.nvidia.com/drivers (610+ recommended) | - |
| NVIDIA own build: `ptxas error : Instruction 'mma with block scale' not supported on .target 'sm_120'` | a hand build for `120` on an RTX 50 | build for `120a` (the setup does) | #40 |
| AMD: `Visual Studio Build Tools (2022 or 2026) are needed` | no C++ workload | the winget command in [4.1](#41-install) | - |
| AMD: `requires HIP 7.14 or newer ... found HIP 7.x` | a system-wide HIP SDK (ROCm 7.2) or `ROCM_PATH` is found first | install the SDK into `.venv` by hand: [4.3](#43-the-rocm-sdk-by-hand) | #43 |
| AMD: the first run sits at `rocm-sdk init` for minutes | it unpacks about 3 GB the first time | wait; later runs only check | #43, #49 |
| AMD: `ROCm's hipInfo sees no AMD GPU` | the driver is old or missing | update Adrenalin, restart | - |
| AMD: CMake's HIP compiler test stops at `__device__ function 'isgreater' cannot overload __host__ __device__ function 'isgreater'` | Visual Studio 2026 (MSVC 14.51) against ROCm SDK 7.14.1's Clang 23 headers | v1.3.131+ fixes the SDK header by itself; check that `ABYSS_NO_ROCM_HEADER_FIX` is not set. When the setup cannot write the file: in `.venv\Lib\site-packages\_rocm_sdk_devel\lib\llvm\lib\clang\23\include\__clang_hip_runtime_wrapper.h` move `#include <__clang_cuda_math_forward_declares.h>` to just above the first `#include <cmath>` | #2, #43 |
| AMD: the build stops in `glm_prefill.cu` with `error: expected unqualified-id` at `float* OUT = M.OUTP;` | `<windows.h>` defines `OUT` as an empty macro, and a v1.3.186 HIP-only function named a variable `OUT` (v1.3.186-v1.3.188) | fixed in v1.3.189: update and run setup again | #49 |
| AMD: hundreds of warnings during the build (`-Wsign-compare`, `-Wunused-function`, ...) | normal | none | #49 |
| AMD: decode is fine, prompts read at a few tok/s (SenseNova 3, Qwen3.8 9 tok/s) | an engine from before v1.3.188 whose MMQ kernels were built at `-O0` | update, run the setup again, check as in [4.4](#44-check-that-the-build-is-optimized) | #3, #43, #49 |
| AMD: very slow everything (~1 tok/s) | every kernel at `-O0` (before v1.3.29) | update | #3 |
| AMD: `--bench` says `the engine stopped during the benchmark`; the log shows a crash in `amdhip64_7.dll` (`hipblasGemmEx` -> `hipblasLtMatmul`) | hipBLASLt in ROCm SDK 7.14.1 | v1.3.182+ (own GEMM) and v1.3.188+ (`ROCBLAS_USE_HIPBLASLT=0` by default). On an older version: `$env:ROCBLAS_USE_HIPBLASLT="0"` before the command | #49 |
| AMD: the start stops at `pinned disk staging did not allocate` | the driver's pinnable memory ran out before the RAM | v1.3.181+ gives RAM-tier slots back | #49 |
| AMD: `--bench`, `--check` or the setup name an old card (e.g. "RX 6600 XT 8 GB" for an RX 9060 XT 16 GB) | Windows' adapter registry keeps removed cards | v1.3.139+ asks ROCm's hipInfo. Display only: the engine measures its own VRAM | #43 |
| AMD: `PCIe ?` in the bench | the link is not readable for an AMD card on Windows | none (the rest of the bench is valid) | #43 |
| `START-ABYSS.bat` without `--config` goes into Maya's GLM-5.3 setup | before v1.3.139 it looked only for Maya's configs | update, or pass `--config .\abyss-<name>.json` | #43 |
| `--bench` measures the old engine after an update | before v1.3.139 it did not rebuild | update; or run the setup once after `git pull` | #43 |
| The log's RAM tier is far below your RAM; many `disk` reads in `glm stat` | the commit limit (page file) | [section 7](#7-ram-the-page-file-and-other-windows-settings) | #4, #43 |
| NVIDIA: large prompts (~20K+) fail with `host buffers did not allocate`, or the engine exits `0xC0000005` | pinned CPU-lane buffers (before v1.3.107) | update; `"STRATA_GLM_PREFILL_CPU": "0"` is an option | #31 |
| DeepSeek V4.1: `an invalid Engram row (NaN code or scale)` | 32-bit file offsets on Windows (before v1.3.34) | update | #6 |
| Everything slow after limiting Abyss to the P-cores in Task Manager | the GPU runtime's threads starve | remove the affinity; use `STRATA_GLM_CPU_PIN` ([section 7](#intel-cpus-with-p-cores-and-e-cores)) | #4, #9, #12 |
| Older build errors: `cudaMemcpyKind`, `M_PI`, `sys/socket.h`, `__builtin_memcpy`, `_cvtsh_ss` | fixed (v1.3.29, v1.3.141, v1.3.120, v1.3.130, v1.3.159) | update | #2, #31, #39 |

## 10. Reporting a result or a problem

Reports from Windows PCs are what this page is built from. Please send them.

- **Speed:** `.\START-ABYSS.bat --config .\abyss-<name>.json --bench` (or `--bench --deep`), then paste
  `abyss-bench.txt` into a [speed report](https://github.com/zephel01/Abyss/issues/new?template=bench.md).
- **A problem:** `.\START-ABYSS.bat --config .\abyss-<name>.json --report`, then open a
  [bug report](https://github.com/zephel01/Abyss/issues/new?template=bug.md) with `abyss-report.txt`, the console
  output and, for a build error, the first `error:` lines of the compiler.
- **A GPU we do not have:** paste `abyss-gpu-tune.txt` into a
  [GPU tune report](https://github.com/zephel01/Abyss/issues/new?template=gpu-tune.md).
- Say: Windows 11, the GPU and driver version, RAM and page file size, the Abyss version, NVIDIA prebuilt or own build
  (and the CUDA version), AMD: the ROCm SDK version and Visual Studio version.

The files have no host name, user name or folder path; read them before you paste.

## 11. What has not been checked on Windows

- The prebuilt NVIDIA engine on Windows (built and checked by CI; no user report yet).
- Whether an NVIDIA own build needs the "x64 Native Tools Command Prompt" (the reporters did not say how they ran it).
- An AMD two-card split, the Ryzen AI Max iGPU alone, RX 9070 / R9700 and the RX 7000 siblings on Windows.
- Antivirus (Microsoft Defender) or SmartScreen prompts, and Windows' long-path limit: nobody has reported a problem.
