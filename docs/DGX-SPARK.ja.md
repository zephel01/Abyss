# NVIDIA DGX Spark（GB10、ARM）- ベータ

[English](DGX-SPARK.md)

v1.3.113 から、Abyss は DGX Spark でビルドして動かせます。DGX Spark は、Grace の ARM CPU（aarch64）と GB10 の GPU
（sm_121）が、128 GB の LPDDR5X を 1 つのメモリとして分け合う機械です。ほかの GB10 の機械（ASUS Ascent GX10 など）も
同じチップですが、まだ動かしていません。

測ったもの: 借りた DGX Spark で 1 日、**Qwen3.8-Flash-Next 16G64G** を動かしました。もっと大きいモデル（DeepSeek V4.1、
DeepSeek V4、GLM-5.3-Flash）は、**本物の Spark ではまだ試していません**（[まだ試していないこと](#まだ試していないこと)）。
`./abyss.sh --bench` の結果を [bench の issue](https://github.com/zephel01/Abyss/issues/new?template=bench.md) で
教えてもらえると助かります。

## Spark で違うところ

- **GPU と CPU が 1 つのメモリを使う。** エンジンは内蔵 GPU（`cudaDevAttrIntegrated`）だと判断し、エキスパートの
  置き場（pool）と pinned の RAM 層を、1 つの予算から決めます。予算は、使えるメモリ（`MemAvailable`。ページキャッシュは
  空きとして数える）から 8 GB（`STRATA_GLM_UMA_HEADROOM_GB`）を引いたものです。`STRATA_GLM_UMA=0` で、外付けのカードと
  同じ決め方になります。
- **programmatic dependent launch（PDL）が有効。** 前のカーネルが終わりかけたところで、次のカーネルを始めます。
  GB10 では、生成の約 1,300 個の小さなカーネルの間のすき間が重く、PDL で +8%（下書きあり）〜 +25%（なし）速く
  なりました。外付けのカードでは変わらないので無効です（`STRATA_GLM_PDL=0/1` でどちらにもできます）。
- **RAM 層のエキスパートは pool の枠に写してから使う**（PC と同じ）。その場で読む方式にはしていません。枠がよく使う
  エキスパートを持ち続けるためで、その場で読む方式では入れ替えが止まりました（32 GB の予算: 写す方式 45.8 tok/s、
  その場で読む方式 34.9）。`STRATA_GLM_UMA_DIRECT=1` がその場で読む方式です（試験用）。
- **CPU レーンは無効。** 20 個の ARM コアがエキスパートを 1 つ計算するより、GPU が同じメモリを読むほうが速いためです
  （`STRATA_GLM_CPU_LANE=<スレッド数>` で有効。Spark では速さは変わりませんでした）。
- **x86 用の CPU カーネルはない。** CPU でのエキスパートの計算は ggml-cpu のもの（NEON。dotprod / i8mm / fp16 は
  `/proc/cpuinfo` を見て使う）です。Q2_0 のエキスパートを持つ pack（DeepSeek V4 の canonical pack）は AVX が要るので
  ARM では動きません。native の pack（`--native`、`iq_pack.py --fast`）は動きます。

## 準備

DGX OS には NVIDIA のドライバーと CUDA 13（`/usr/local/cuda`）が入っています。sm_121 には CUDA 12.9 以上が要ります。

1. **ソース、llama.cpp、Python のパッケージ**（llama.cpp はビルドが決めているコミットにする）:

   ```sh
   git clone https://github.com/zephel01/Abyss.git && cd Abyss
   git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
   git -C third_party/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
   sudo apt-get install -y cmake ninja-build python3-venv
   python3 -m venv .venv
   .venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil huggingface_hub sentencepiece
   ```

2. **エンジン**（Spark の 20 コアで約 1 分）:

   ```sh
   cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
     -DCMAKE_CUDA_ARCHITECTURES=121a -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
     -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_IGPU_LANE=OFF \
     -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
   cmake --build build-cuda --target strata abyss_gpu_tune -j"$(nproc)"
   ./build-cuda/strata --help > /dev/null && echo OK
   ```

   `./setup.sh`（インストーラー）にも ARM 用の直しは入っていますが、本物の Spark ではまだ動かしていません。
   上のコマンドが、実際に使ったものです。

3. **モデルと pack**（Qwen3.8 16G64G、3 つのファイルで 86.9 GB）:

   ```sh
   .venv/bin/hf download zephel01/Qwen3.8-Flash-Next-16G64G-GGUF --local-dir ~/models/q38-16g64g
   (cd ~/models/q38-16g64g && sha256sum -c SHA256SUMS)
   PYTHONPATH=third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py --fast \
     --gguf ~/models/q38-16g64g/Qwen3.8-Flash-Next-16G64G-00001-of-00003.gguf --out ~/models/q38-16g64g/pack-fast
   ```

   MTP の下書き器（おすすめ: 31 → 47 tok/s）は、[docs/qwen38](qwen38/README.ja.md) の準備の手順 2 と同じように作ります。

4. **設定ファイル**。Abyss のフォルダーに `abyss-q38-16g64g.json` として保存します（`/home/you` は自分のホームフォルダー）:

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

   起動は `.venv/bin/python serve/server.py --engine strata --config abyss-q38-16g64g.json --port 8082` です。

5. **確かめる・測る**（このあいだはエンジンを止めておく）:

   ```sh
   ./abyss.sh --check          # この PC とモデル: 日本語、コード、長いプロンプト、長い会話（15〜30 分）
   ./abyss.sh --bench --deep   # 速さと、1 トークンの時間がどこで使われているか
   ./abyss.sh --gpu-tune       # この GPU でのカーネルの選び方
   ./abyss.sh --report         # issue に貼る abyss-report.txt（パスや名前は消してある）
   ```

   Spark では、PC で PCIe のつなぎ方が出るところに「memory unified」と出ます。

## 測った数字（借りた DGX Spark、2026-11-05）

GB10、共有メモリ 121 GB、Qwen3.8-Flash-Next 16G64G、既定の予算（エキスパートが全部メモリに入る）、greedy。

**生成（CLI、4K のプロンプト、512 トークンを 3 回の中央値）:**

| | なし | graph だけ | PDL だけ | graph + PDL（既定） |
|---|---:|---:|---:|---:|
| MTP の下書きあり | 43.40 | 44.58 | 46.51 | **47.05 tok/s（+8.4%）** |
| なし | 25.01 | 25.07 | 30.73 | **31.20 tok/s（+25%）** |

- 32K のプロンプト: 下書きありは 40.8 → 44.05 tok/s、なしは 24.6 → 29.6。
- graph / PDL のあり・なしで、出てくるトークンは同じ。NVFP4 版との KL（seq 9）は、どの設定でも 0.170186。
- サーバー経由（実際の使い方、下書きあり）: 日本語の 5 問とコードの 5 題で 43〜54 tok/s、12 往復の会話で
  36.6〜56 tok/s。`./abyss.sh --bench` は平均 53.7（256 トークンの答え 3 つ）。

**プロンプト（CLI）:**

| 長さ | 8K | 24K | 32K | 64K | 128K | 262K |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 660 | 787 | 746 | 779 | 799 | 612 |

`--bench`: 2K で 886、8K で **約 1,040 tok/s**。24K、128K、262K のプロンプトは、下書きありで最後まで動きました。

**実際の作業**（サーバー、既定の設定）: 日本語の 5 問（敬語のメール、説明、要約、知識と計算の 2 問）が正解、コードは
5 題ともテストが通り、8K のプロンプトの中の needle も見つけました。12 往復の会話も最後まで動き、毎回プロンプトの
使い回しが効きました（1 回 0.26〜0.63 秒）。

**予算を小さくしたとき**（`STRATA_GLM_VRAM_GB=32`、PC と同じくらいの 32 GB の置き場）: 下書きありで 45.8 tok/s
（16 GB では 42.5）。v1.3.113 では、このときのプロンプトの道を直しました。上限を付けると、ページキャッシュを「使用中」
と数えて、プロンプトを 1 トークンずつ読んでいました（3.8K トークン: 129 秒 → 6.5 秒）。

## SSD の層

メモリより大きいモデルは、答えている間に、残りのエキスパートを SSD から読みます。借りた機械のディスクはループデバイスで
（1 本で 0.98 GB/s、8 本で 1.5 GB/s）、Spark の NVMe（約 13 GB/s）とはかけ離れています。下の数字は Spark の数字では
なく、傾向だけを見るためのものです:

| 置き場の予算（残りは SSD から） | 既定（`STRATA_GLM_READ_CHUNKS` 8） | `STRATA_GLM_READ_CHUNKS=1` |
|---|---:|---:|
| 12 GB | 7.7 tok/s | 11.0 |
| 20 GB | 13.0 | 17.2 |
| 28 GB | 20.4 | 24.6 |

- この遅いディスクでは、細かく何度も読むより、大きく少なく読むほうが速くなりました（`STRATA_GLM_READ_CHUNKS=1` で
  +21〜43%）。既定の 8 は PC の NVMe で決めた値なので、変えていません。本物の Spark では両方を試してください。
- 1 トークンの時間は、どの点でも「21 ms + SSD の待ち」に合いました。同じ式で速いディスクを見積もると（見積もり）:
  12 / 20 / 28 GB で、7 GB/s なら約 13 / 22 / 31 tok/s、13.4 GB/s なら約 20 / 28 / 36 tok/s。
- 読み込みは O_DIRECT なので、ページキャッシュとメモリを取り合いません（空きメモリは 83〜99 GB のまま）。

## まだ試していないこと

- 本物の Spark での DeepSeek V4.1 / V4 と GLM-5.3-Flash（PDL、共有メモリでの大きさの決め方、写す方式は同じコードなので
  効くはずですが、測っていません）。
- Spark の NVMe を SSD の層に使うこと（上のとおり、借りた機械のディスクはループデバイスでした）。
- ARM での `./setup.sh`（ビルドは [準備](#準備) のとおり手で行いました）。
- Spark 2 台をつなぐこと、ASUS Ascent GX10。
