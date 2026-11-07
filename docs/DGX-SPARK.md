# NVIDIA DGX Spark (GB10, ARM) - beta

[日本語](DGX-SPARK.ja.md)

From v1.3.113 Abyss builds and runs on the DGX Spark: a Grace ARM CPU (aarch64) and a GB10 GPU (sm_121) sharing one
128 GB LPDDR5X memory. Other GB10 boxes (ASUS Ascent GX10 and the like) are the same chip; we have not run one.

What is measured: **Qwen3.8-Flash-Next 16G64G** on a rented DGX Spark (2026-11-05), then **GLM-5.3-Flash, DeepSeek V4
0731 and DeepSeek V4.1** on a rented DGX Spark with v1.3.114 (2026-11-06, [below](#measured-big-models-rented-dgx-spark-2026-11-06)).
Reports are welcome: [a bench issue](https://github.com/zephel01/Abyss/issues/new?template=bench.md) with `./abyss.sh --bench`.

## What is different on a Spark

- **One memory for the GPU and the CPU.** The engine sees an integrated GPU (`cudaDevAttrIntegrated`) and sizes the
  expert pool and the pinned RAM tier from ONE budget: the memory available (`MemAvailable`, page cache counted as
  free) less a headroom of 8 GB (`STRATA_GLM_UMA_HEADROOM_GB`). `STRATA_GLM_UMA=0` uses the discrete-card rules.
- **Programmatic dependent launch (PDL) is on.** The next kernel starts while the one before it finishes. On a GB10
  the gaps between the decode's ~1,300 small kernels are the cost; PDL takes +8% (with the drafter) to +25% (without).
  Off on discrete cards, where it changes nothing (`STRATA_GLM_PDL=0/1` sets it either way).
- **RAM-tier experts are copied into the pool's slots** (as on a PC), not read in place: the slots keep the hot
  experts, and the reads in place stopped the swaps (32 GB budget: copy 45.8 tok/s, in place 34.9).
  `STRATA_GLM_UMA_DIRECT=1` is the in-place mode, for tests.
- **The CPU lane is off.** The GPU reads the same memory faster than 20 ARM cores compute an expert
  (`STRATA_GLM_CPU_LANE=<threads>` turns it on; on the Spark it did not change the speed).
- **No x86 CPU kernels.** The CPU's expert dots are ggml-cpu's own (NEON, dotprod / i8mm / fp16 from
  `/proc/cpuinfo`). Packs with Q2_0 experts (the canonical DeepSeek V4 pack) need AVX and do not run on ARM; the
  native packs (`--native`, `iq_pack.py --fast`) do.
- **No start on memory another process holds** (v1.3.115). When other processes hold more than 16 GB of the shared
  memory and the experts no longer fit because of it, the engine stops at load and names the biggest ones (a second
  engine left running had made one start on 20 GB of free memory, and the machine ran out of memory).
  `STRATA_GLM_UMA_SHARE=1` starts anyway, `STRATA_GLM_UMA_OTHERS_GB` sets the limit; a model bigger than the machine
  with nothing else running is not stopped.

## Setup

DGX OS ships the NVIDIA driver and CUDA 13 (`/usr/local/cuda`). CUDA 12.9 or newer is needed for sm_121.

1. **Source, llama.cpp and Python packages** (llama.cpp at the commit the build pins):

   ```sh
   git clone https://github.com/zephel01/Abyss.git && cd Abyss
   git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
   git -C third_party/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
   sudo apt-get install -y cmake ninja-build python3-venv
   python3 -m venv .venv
   .venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil huggingface_hub sentencepiece
   ```

2. **The engine** (about a minute on the Spark's 20 cores):

   ```sh
   cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
     -DCMAKE_CUDA_ARCHITECTURES=121a -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
     -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_IGPU_LANE=OFF \
     -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
   cmake --build build-cuda --target strata abyss_gpu_tune -j"$(nproc)"
   ./build-cuda/strata --help > /dev/null && echo OK
   ```

   `./setup.sh` (the installer) has the ARM changes too, but it has not been run on a real Spark; the commands
   above are what we ran.

3. **The model and its pack** (Qwen3.8 16G64G, 86.9 GB in three shards):

   ```sh
   .venv/bin/hf download zephel01/Qwen3.8-Flash-Next-16G64G-GGUF --local-dir ~/models/q38-16g64g
   (cd ~/models/q38-16g64g && sha256sum -c SHA256SUMS)
   PYTHONPATH=third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py --fast \
     --gguf ~/models/q38-16g64g/Qwen3.8-Flash-Next-16G64G-00001-of-00003.gguf --out ~/models/q38-16g64g/pack-fast
   ```

   The MTP drafter (recommended: 31 -> 47 tok/s) is made as in [docs/qwen38](qwen38/README.md#setup), step 2.

4. **The config**, saved as `abyss-q38-16g64g.json` in the Abyss folder (`/home/you` is your home folder):

   ```json
   {
    "exe": "build-cuda/strata",
    "args": ["--glm-pack", "/home/you/models/q38-16g64g/pack-fast", "--max-context", "32768", "--prefill", "auto"],
    "tokenizer": "/home/you/models/q38-16g64g/pack-fast/tokenizer",
    "model_name": "qwen3.8-flash-next-16g64g",
    "gpu": [0],
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "lib_dirs": ["/usr/local/cuda/lib64"],
    "port": 8082,
    "backend": "cuda",
    "env": {"STRATA_Q4X_MTP": "/home/you/models/qwen38-mtp/rt"}
   }
   ```

   Start it with `.venv/bin/python serve/server.py --engine strata --config abyss-q38-16g64g.json --port 8082`.

5. **Check and measure** (the engine must be stopped while these run):

   ```sh
   ./abyss.sh --check          # this PC and the model: Japanese, code, long prompts, a long conversation (15-30 min)
   ./abyss.sh --bench --deep   # speed and where a token's time goes
   ./abyss.sh --gpu-tune       # the GPU's kernel choices
   ./abyss.sh --report         # abyss-report.txt for an issue (paths and names removed)
   ```

   On a Spark the reports say "memory unified" where a PC's say the PCIe link. `--gpu-tune` finds the hand-built
   `build-cuda` engine through the config's `exe` from v1.3.115 (before, it stopped with "not built here yet"; #37),
   and `--check`'s Japanese arithmetic task no longer fails a short right answer from v1.3.114 (#37).

## Measured (rented DGX Spark, 2026-11-05)

GB10, 121 GB unified memory, Qwen3.8-Flash-Next 16G64G, the default budget (every expert fits in memory), greedy.

**Answers (CLI, 4K prompt, median of 3 x 512 tokens):**

| | nothing | graphs only | PDL only | graphs + PDL (the default) |
|---|---:|---:|---:|---:|
| with the MTP drafter | 43.40 | 44.58 | 46.51 | **47.05 tok/s (+8.4%)** |
| without | 25.01 | 25.07 | 30.73 | **31.20 tok/s (+25%)** |

- 32K prompt: 40.8 -> 44.05 tok/s with the drafter, 24.6 -> 29.6 without.
- The same tokens with and without graphs / PDL; KL to the NVFP4 teacher (seq 9) 0.170186 in every setting.
- Through the server (real use, the drafter on): 43-54 tok/s over five Japanese questions and five code tasks,
  36.6-56 over a 12-turn conversation; `./abyss.sh --bench`: mean 53.7 (three answers of 256 tokens).

**Prompts (CLI):**

| length | 8K | 24K | 32K | 64K | 128K | 262K |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 660 | 787 | 746 | 779 | 799 | 612 |

`--bench`: 2K 886, 8K **~1,040 tok/s**. 24K, 128K and 262K prompts with the drafter ran to the end.

**Real use** (server, defaults): five Japanese questions right (a polite e-mail, an explanation, a summary, two
fact / arithmetic questions), code 5 of 5 tests passing, the needle in an 8K prompt found, a 12-turn conversation to the
end with each turn's prompt reused (0.26-0.63 s a turn).

**A smaller budget** (`STRATA_GLM_VRAM_GB=32`, a PC-like 32 GB pool): 45.8 tok/s with the drafter (16 GB: 42.5).
v1.3.113 fixes the prompt path there: under a cap it counted the page cache as used and read prompts token by token
(3.8K tokens: 129 s -> 6.5 s).

## Measured: big models (rented DGX Spark, 2026-11-06)

A rented DGX Spark (128 GB unified memory), Abyss v1.3.114, the default budget.

| model | decode | 8K prompt | notes |
|---|---:|---:|---|
| GLM-5.3-Flash (Maya-S-v2 IQ2_XXS, unsloth UD-IQ2_XXS) | 19-20 tok/s | 730-750 tok/s | 32K prompt from the CLI 675-796 tok/s; the MTP drafter gave no gain until v1.3.118 (below) |
| GLM-5.3-Flash, v1.3.118 (one-GPU verify window + NextN drafter, on by default here) | **S1 (zephel01, 94 GB) 24.3 tok/s** (21.4 without), **UD-IQ2_XXS 22.9** (19.8 without), same tokens | | 4K prompt, median of 3 x 512 tokens; drafts accepted 74-80% |
| GLM-5.3-Flash, v1.3.119 (faster verify window) | **S1 27.4 tok/s**, **UD-IQ2_XXS 24.2** with the window changes alone (the same tokens as v1.3.118); S1 28.6 with the whole release | | 4K prompt, median of 3 x 512 tokens; the whole release changes greedy tokens here (the prompt tail on top: `STRATA_GLM_TAIL_ON_TOP`) |
| DeepSeek V4 0731 | 21.0 tok/s, **22.8 with the DSpark drafter** (78% accepted) | 814 tok/s | |
| DeepSeek V4 0731, v1.3.119 | **25.2 tok/s with the DSpark drafter** with the window change alone (23.7 with v1.3.118 the same day, the same tokens); 26.0 with the whole release | | 4K prompt, median of 3 x 512 tokens |
| Qwen3.8 ISTA IQ3_XXS, v1.3.119 (with the MTP drafter) | **55.8 tok/s after a 4K prompt, 53.4 after 32K** (v1.3.118 the same day: 54.7 / 47.5) | | median of 3 x 512 tokens; the decode after a long prompt no longer re-reads experts from the SSD |
| DeepSeek V4.1 Compact (zephel01's 207 GB build, does not fit in memory) | ~18 tok/s | | |

**llama.cpp on the same Spark** (8K prompt, 4K context filled): V4 0731 decode 20.2 tok/s, prompt 360 tok/s;
GLM-5.3-Flash 19.3 / 321. Abyss decodes the same or faster and reads prompts ~2.3x faster.

**Known issue (open, seen only on the Spark):** GLM-5.3-Flash's 32K-121K prompts through the server run at ~120-130
tok/s, while the CLI reads them at 675-796. Being investigated; an RTX 5090 does not show it.

## The SSD tier

Models bigger than the memory read the rest of their experts from the SSD while answering. The rented machine's disk
was a loop device (0.98 GB/s one reader, 1.5 GB/s eight), far from a Spark's own NVMe (~13 GB/s), so these are not a
Spark's numbers - only the shape:

| pool budget (rest from the SSD) | default (`STRATA_GLM_READ_CHUNKS` 8) | `STRATA_GLM_READ_CHUNKS=1` |
|---|---:|---:|
| 12 GB | 7.7 tok/s | 11.0 |
| 20 GB | 13.0 | 17.2 |
| 28 GB | 20.4 | 24.6 |

- On that slow disk a few large reads beat many small ones (`STRATA_GLM_READ_CHUNKS=1`, +21-43%). The default (8)
  was chosen on a PC's NVMe and stays; on a real Spark try both.
- A token's time fitted `21 ms + the SSD waits` at every point. The same fit for a faster disk, an estimate:
  12 / 20 / 28 GB -> about 13 / 22 / 31 tok/s at 7 GB/s and 20 / 28 / 36 at 13.4 GB/s.
- Reads go with O_DIRECT, so they do not fight the page cache for memory (free memory stayed at 83-99 GB).

## Not tested yet

- A Spark's own NVMe as the SSD tier (above: the rented disk was a loop device).
- `./setup.sh` on ARM (the build was done by hand, as in [Setup](#setup)).
- Two Sparks together, and the ASUS Ascent GX10.
