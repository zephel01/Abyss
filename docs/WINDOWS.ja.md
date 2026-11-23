# Windows 11 で Abyss を使う（NVIDIA と AMD）

[English](WINDOWS.md)

Windows 11 に Abyss を入れて動かす手順です。NVIDIA の GeForce と AMD の Radeon の両方を扱います。v1.3.189 の時点の内容です。

**最初に読んでください。** Abyss の開発と確認は Linux で行っています。こちらには Windows の PC がないため、このページの手順をこちらで
Windows 上で実行したことはありません。手順はセットアップのコードと、利用者が GitHub の issue に書いてくれた報告をもとにしています。
報告はそれぞれ issue 番号（#番号）でリンクしています。まだ誰も報告していない手順には **報告なし** と書いています。

## 目次

- [1. 必要なもの](#1-必要なもの)
- [2. セットアップの前に](#2-セットアップの前に)
- [3. NVIDIA](#3-nvidia)
- [4. AMD Radeon](#4-amd-radeon)
- [5. PC に合うモデル](#5-pc-に合うモデル)
- [6. 起動と計測](#6-起動と計測)
- [7. RAM とページファイルと Windows の設定](#7-ram-とページファイルと-windows-の設定)
- [8. 更新](#8-更新)
- [9. トラブル対応](#9-トラブル対応)
- [10. 報告のしかた](#10-報告のしかた)
- [11. Windows で確かめられていないこと](#11-windows-で確かめられていないこと)

## 1. 必要なもの

| | NVIDIA | AMD |
| --- | --- | --- |
| Windows | Windows 11（64 ビット） | Windows 11（64 ビット） |
| GPU | ビルド済みエンジンは RTX 30 / 40 / 50。それ以外（RTX 20、V100 など）は自分でビルド | RX 7900 XT / XTX（gfx1100）、RX 9070 / 9070 XT / AI PRO R9700（gfx1201）、RX 9060 / 9060 XT（gfx1200）、Ryzen AI Max の Radeon 8050S / 8060S（gfx1151）。同系統の RX 7800 XT / 7700 XT（gfx1101）、RX 7600（gfx1102）、Radeon 890M（gfx1150）はビルドできるが未検証。RDNA2（RX 6000）以前は対象外 |
| ドライバ | ビルド済みエンジンは **580 以上が必須**、610 以上を推奨 | AMD Software: Adrenalin Edition（https://www.amd.com/en/support） |
| Visual Studio Build Tools | ビルド済みエンジンなら **不要**。自分でビルドするなら 2022（または 2019）の「C++ によるデスクトップ開発」。CUDA 13 は Visual Studio 2026 を受け付けない | **必要**。2022 か 2026 の「C++ によるデスクトップ開発」（ROCm のコンパイラが、そのヘッダとライブラリを使う） |
| CUDA / ROCm | ビルド済みエンジンなら **不要**（CUDA 13.3 のライブラリが同梱）。自分でビルドするなら CUDA Toolkit 12.8 以上 | AMD の ROCm SDK 7.14.1（TheRock）。セットアップが Abyss 専用の `.venv` に入れる |
| Python | 64 ビット、3.10 以上 | 同じ |
| CPU | AVX2 | AVX2（無いと AMD のセットアップは止まる） |

これまでに Windows で報告があった構成:

| カード | 報告 |
| --- | --- |
| RTX 4090 24 GB + RAM 96 GB、自分でビルド（MSVC + CUDA） | DeepSeek V4 Flash 0731（#31、#46）、Qwen3.8-Flash-Next UD-Q4_K_XL（#39） |
| RTX 5060 Ti 16 GB × 2 + RAM 128 GB、自分でビルド、CUDA 13.3 | GLM-5.3-Flash（#40） |
| RX 7900 XT 20 GB + RAM 128 GB、ROCm SDK 7.14.1 | DeepSeek V4 0731、V4.1（#2〜#6、#9、#12） |
| RX 9060 XT 16 GB + RAM 64 GB、ROCm SDK 7.14.1、VS 2026 Build Tools（MSVC 14.51）、Adrenalin 26.11.2。v1.3.189 でビルドと実行を確認（MMQ が最適化してビルドされた） | Ornith-1.5-122B、Qwen3.8-Flash-Next 12G32G / 16G64G、SenseNova-Loop-9B（#43、#47、#49） |

Windows での NVIDIA のビルド済みエンジン: **報告なし**。

### ディスクの空き

- モデル: 9B のモデルの 5.6 GB から、16G64G〜96G の各サイズの 100 GB 前後まで。DeepSeek V4.1 は 207〜283 GB です。モデルごとの
  大きさは [README](../README.ja.md) にあります。
- そのほかに、エンジンのビルド、llama.cpp のソース、Python のパッケージで約 3 GB。
- AMD は `.venv` に ROCm SDK が入ります。初回の `rocm-sdk init` で約 3 GB が展開されます。
- モデルは NVMe の SSD に置いてください。VRAM と RAM に収まらないモデルでは、SSD の速さがそのまま生成の速さになります。

### RAM とページファイル

Windows は、エキスパートを置く固定（pinned）の RAM（「RAM 段」）を **コミットの上限**（RAM + ページファイル）で数えます。
ページファイルが小さいと、RAM が空いていてもエキスパートが SSD から読まれます。最初に動かす前に、ページファイルを設定しておいてください:
[7 章](#7-ram-とページファイルと-windows-の設定)。

## 2. セットアップの前に

コマンドはすべて **PowerShell**（Windows ターミナル、または「Windows PowerShell」）用です。ふだんのユーザーで実行してください。
セットアップに管理者のウィンドウは要りません。

1. **Python**（64 ビット、3.10 以上）。Python が無いときは、セットアップの `.bat` も同じコマンドを表示します。

   ```powershell
   winget install -e --id Python.Python.3.12 --scope user
   ```

   https://www.python.org/downloads/ のインストーラを使う場合は、「Add python.exe to PATH」にチェックを入れてください。
   入れたあとは、PowerShell のウィンドウを開き直します。

2. **Git。** 更新（ダッシュボードの About > Update、または `git pull`）には git で取ったフォルダが必要です。ZIP でダウンロードした
   フォルダは、自分では更新できません。

   ```powershell
   winget install -e --id Git.Git
   ```

3. **Abyss を取ってくる。** 場所は、空白を含まない短いパスが安全です。HIP のビルドは、パスに空白が無いときだけ渡すコンパイラの設定が
   あるためです。報告者は `C:\AI\...`、`D:\AI\...`、`C:\Users\<名前>\...` を使っていました。

   ```powershell
   mkdir C:\AI -Force
   cd C:\AI
   git clone https://github.com/zephel01/Abyss.git
   cd C:\AI\Abyss
   ```

   モデルは `C:\AI\Abyss-data` に、ツールのフォルダは `C:\AI\Abyss-tools` に入ります（Abyss の隣）。
   モデルを別のドライブに置くときは `--data-dir D:\Abyss-data` を付けます。

4. **ページファイル**: [7 章](#7-ram-とページファイルと-windows-の設定)。再起動が要るので、先に済ませておきます。

セットアップのファイルは 2 つあります。

| ファイル | 何をするか |
| --- | --- |
| `setup-deepseek.bat` | README の表にあるモデルを入れる（DeepSeek V4 / V4.1、Qwen3.8、Ornith、`--model glm` で GLM-5.3-Flash、9B のモデルなど）。中身は `tools\abyss_setup.py`。**このページではこちらを使います** |
| `START-ABYSS.bat` | 入れたモデルを起動する（`--config`）。`--bench`、`--calibrate`、`--check`、`--report`、`--gpu-tune` もこちら。config を指定しないと、Maya の GLM-5.3-Flash のインストーラになる |

どちらも、初回に Abyss のフォルダの中に専用の Python 環境 `.venv` を作ります。システムには何も入れません。
`.\setup-deepseek.bat --help` で、すべてのオプションが出ます。

## 3. NVIDIA

### 3.1 ビルド済みエンジン

CUDA Toolkit（`nvcc`）も Visual Studio も要らない方法です。v1.3.164 から、セットアップは `nvcc` か Visual Studio の C++ コンパイラが
見つからないと、GitHub Actions でビルドしたエンジンをダウンロードします。Windows ではこれがいちばん簡単です。

| | |
| --- | --- |
| ファイル | この版の GitHub リリース（無ければ最新）の `abyss-engine-cuda-windows-x64.zip`。リリースの `SHA256SUMS` と照合して `build\` に展開 |
| GPU | RTX 30（sm_86）、RTX 40（sm_89）、RTX 50（sm_120a）を 1 つのエンジンで。ほかのカードは理由を出して止まる |
| 中身 | `abyss.exe`、`abyss_gpu_tune.exe`、NVIDIA の cuBLAS / cuBLASLt の DLL（CUDA 13.3） |
| ドライバ | 580 以上が必須（それより古いとセットアップが止まる）、610 以上を推奨 |
| まだできないこと | 画像。画像のエンコーダはまだ CUDA Toolkit でのビルドが要るので、ビルド済みエンジンではテキストだけ |

1. ドライバの版を確かめます。

   ```powershell
   nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
   ```

   580 より古ければ、https://www.nvidia.com/drivers から更新して再起動します。

2. PC に合うモデルを指定してセットアップを実行します（[5 章](#5-pc-に合うモデル)）。たとえば 16 GB のカード + RAM 64 GB なら:

   ```powershell
   cd C:\AI\Abyss
   .\setup-deepseek.bat --model qwen38 --download-model
   ```

   PC の確認、Python のパッケージ、ビルド済みエンジンのダウンロード、モデルのダウンロード（全ファイルを Hugging Face の SHA-256 と照合）、
   パックの作成、config `abyss-<名前>.json` の書き出し、GPU のカーネルの選び方の計測（`--gpu-tune`、1 分ほど）を順に行います。
   済んだ手順は飛ばすので、途中で止まっても、もう一度実行すれば続きから進みます。

3. CUDA Toolkit と Visual Studio が入っている PC では、セットアップはビルドする方を選びます。それでもビルド済みエンジンを使うときは:

   ```powershell
   .\setup-deepseek.bat --model qwen38 --download-model --prebuilt
   ```

このあとは [6 章](#6-起動と計測) へ進みます。

### 3.2 エンジンを自分でビルドする

ビルド済みエンジンが対応していないカード、画像を使いたいとき、自分のチェックアウトからビルドしたいときの方法です。
RTX 4090（#31、#39、#46）と RTX 5060 Ti（#40）で、利用者がこの方法でビルドしています。

1. **Visual Studio 2022 Build Tools** の「C++ によるデスクトップ開発」。CUDA 13 が受け付けるのは Visual Studio 2019 と 2022 だけです。
   CUDA のビルドでは、セットアップは Visual Studio 2026 を選びません。

   ```powershell
   winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

2. **Windows 用の CUDA Toolkit** の 12.8 以上（RTX 50 は 12.8 以上が必要。#40 は 13.3 でビルド）:
   https://developer.nvidia.com/cuda-downloads 。セットアップは `nvcc` を、PATH、`CUDA_PATH`、
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*` の順に探します。

3. 3.1 と同じコマンドを、`--prebuilt` なしで実行します。カードに合わせて `build\abyss.exe` をビルドします。RTX 50 は `120a` になります
   （`120` ではありません。MXFP4 のカーネルに `120a` が要ります。#40）。

   ビルドが C++ コンパイラを見つけられないときは、PowerShell ではなくスタートメニューの「x64 Native Tools Command Prompt for VS 2022」
   から実行してください（`cd /d C:\AI\Abyss` のあと `setup-deepseek.bat ...`）。（これが必要かどうかは、こちらでは確かめられていません。）

以前の版は、Windows でビルドするのにローカルの修正が要りました。いまはすべて本体に入っています（#31: v1.3.120、#39: v1.3.130、
#46: v1.3.147 で「ローカルの修正はゼロ」）。リリースのたびに、CI が Windows の CUDA エンジンをビルドしています。

## 4. AMD Radeon

HIP のエンジンは試験的で、テキストだけを扱います。Windows では 1 枚の単体カード（または Ryzen AI Max の iGPU だけ）で動きます。
Linux の「iGPU レーン」（単体の Radeon と APU の iGPU を一緒に使う方法）は、Windows にはありません。

### 4.1 インストール

1. **ドライバ:** AMD Software: Adrenalin Edition（https://www.amd.com/en/support）。入れたら再起動します。
2. **Visual Studio Build Tools** の「C++ によるデスクトップ開発」。セットアップが表示するコマンドは次のとおりです。

   ```powershell
   winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   ```

   v1.3.131 からは Visual Studio 2026（MSVC 14.51）でも動きます（#43: RX 9060 XT で VS 2026 Build Tools）。両方が入っていると、
   セットアップは 2022 を使います。
3. **Python と Git:** [2 章](#2-セットアップの前に)。
4. **`--backend hip` と、PC に合うモデルを付けてセットアップを実行します。**

   ```powershell
   cd C:\AI\Abyss
   .\setup-deepseek.bat --backend hip --model ornith --download-model
   ```

### 4.2 AMD のセットアップが自動で行うこと

- **AMD の ROCm SDK を `.venv` に入れる。** 先に確認してから、repo.amd.com から pip で入れます。中身は ROCm SDK 7.14.1（TheRock の
  wheel）と、カードのアーキテクチャ用のカーネルです。システムには何も入れません。
- **`rocm-sdk init`。** 初回は、コンパイラ、ヘッダ、ライブラリを展開します（約 3 GB、数分）。画面には
  `AMD's ROCm SDK: rocm-sdk init (the first time it unpacks about 3 GB) ...` と出ます。2 回目からは確認だけです。
- **Visual Studio 2026 向けに SDK のヘッダを 1 か所直す。** MSVC 14.50 以降と組み合わせると、ROCm 7.14.1 の Clang 23 は
  `'isgreater' cannot overload` で止まります（#2、#43）。セットアップは `.venv` 内の SDK の `__clang_hip_runtime_wrapper.h` で
  `#include` を 1 行移します（LLVM 自身の修正 llvm-project 01d3932 と同じ内容。Windows 用の SDK にはまだ入っていません）。1 回だけ行い、
  元のファイルは `__clang_hip_runtime_wrapper.h.abyss-orig` として残します。`ABYSS_NO_ROCM_HEADER_FIX=1` なら SDK には触りません。
  v1.3.131 から。
- **`build-hip\abyss.exe` をビルドする。** Ninja と ROCm の clang で、gfx1100、gfx1201、gfx1151 と、お使いのカードのアーキテクチャ向けに
  ビルドします（#43 では `gfx1100;gfx1201;gfx1151;gfx1200`）。初回は時間がかかります。警告がたくさん出ますが（#49）、問題ありません。
  ROCm の `amdhip64*.dll` をエンジンの隣にコピーするので、エンジンは System32 にあるドライバ側のものを読みません。
- **`-O3` でビルドする。** Windows の一部の環境（ROCm SDK 7.14.1 + CMake + Ninja）では、CMake の Release 用の HIP のフラグが空に
  なります。v1.3.29 より前は、そのせいで全カーネルが `-O0`（最適化なし）でビルドされ、約 46 tok/s のところが 1.3 tok/s でした（#3）。
  v1.3.188 より前も、プロンプト用の MMQ カーネルだけは `-O0` のままで、プロンプトが 100 倍以上遅くなっていました（#49）。
  v1.3.188 からは、すべての HIP カーネルに `-O3 -DNDEBUG` が付きます。
- **config を書く。** `"backend": "hip"` と、`env` に次の値を入れます: `STRATA_GLM_RAM_HEADROOM_GB` 16、`GPU_MAX_HW_QUEUES` 2、
  `STRATA_GLM_RESERVE_MB` 3072（画面をつないでいるカード）、`STRATA_GLM_SPLIT` 0（カード 1 枚）。あわせて、#4 で見つかった Windows 向けの
  調整を表示します（[7 章](#7-ram-とページファイルと-windows-の設定)）。
- **hipBLASLt を使わない。** ROCm SDK 7.14.1 の hipBLASLt は、`amdhip64_7.dll` の中で落ちました（#49）。v1.3.182 からはプロンプトの
  行列積をエンジン自前のカーネルで計算し、v1.3.188 からは、Windows ではサーバーと `--bench` が `ROCBLAS_USE_HIPBLASLT=0` を付けて
  エンジンを起動します。シェルか config の `"env"` に値があれば、そちらが優先です。
- **`--gpu-tune` を実行する**（1 分ほど）。結果は `abyss-gpu-tune.txt` に書きます。

### 4.3 ROCm SDK を手で入れる

セットアップが SDK を入れるのは、ROCm が **1 つも** 見つからないときだけです。古い ROCm が見つかると、入れずに止まります。

- **AMD の HIP SDK for Windows（ROCm 7.x）がシステムに入っている**（`C:\Program Files\AMD\ROCm\...`、`HIP_PATH`）: セットアップは
  `requires HIP 7.14 or newer` で止まります（#43: ROCm 7.2 が入っていた）。
- **`ROCM_PATH` が古い ROCm を指している**: 同じように止まります。

対処: Abyss の `.venv` に SDK を手で入れます。セットアップは、`HIP_PATH` や `C:\Program Files\AMD\ROCm` より先に `.venv` の SDK を
使います（`.venv` より先に見るのは `ROCM_PATH` だけです）。確かめてから入れます（コマンドは #43 のもの）:

```powershell
cd C:\AI\Abyss
$env:ROCM_PATH; $env:HIP_PATH        # いま設定されている値（空行なら未設定）
Remove-Item Env:ROCM_PATH -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel,device-gfx1200]==7.14.1"
.\setup-deepseek.bat --backend hip --model ornith --download-model
```

`.venv` は、`setup-deepseek.bat` を 1 回実行すればできています。`Remove-Item` で消えるのは、このウィンドウの `ROCM_PATH` だけです。
ずっと消すには「環境変数」の画面（`rundll32 sysdm.cpl,EditEnvironmentVariables`）で削除してください。
システムの HIP SDK をアンインストールする必要はありません。

`device-gfx1200` は、お使いのカードのアーキテクチャに置き換えてください。カードが複数なら、カンマで区切って並べます
（`device-gfx1100,device-gfx1201`）。

| カード | アーキテクチャ | pip の指定 |
| --- | --- | --- |
| RX 7900 XT / XTX | gfx1100 | `device-gfx1100` |
| RX 7800 XT / 7700 XT | gfx1101 | `device-gfx1101` |
| RX 7600 | gfx1102 | `device-gfx1102` |
| RX 9070 / 9070 XT、AI PRO R9700 | gfx1201 | `device-gfx1201` |
| RX 9060 / 9060 XT | gfx1200 | `device-gfx1200` |
| Radeon 8050S / 8060S（Ryzen AI Max） | gfx1151 | `device-gfx1151` |
| Radeon 890M | gfx1150 | `device-gfx1150` |

（#2 は古い書き方の `rocm[libraries,devel,gfx1100]` を使っていました。いまのセットアップは `device-<arch>` です。）

### 4.4 ビルドが最適化されているか確かめる

一度動かすと、エンジンのログと `--bench` に `MMQ kernels built optimized` と出ます。`built WITHOUT optimization (-O0)` と出たら
v1.3.188 より前のエンジンです。更新して、セットアップをもう一度実行してください（[8 章](#8-更新)）。手で確かめるなら（docs/amd と同じ）:

```powershell
cd C:\AI\Abyss
Select-String -Path build-hip\CMakeCache.txt -Pattern "^CMAKE_HIP_FLAGS"
Select-String -Path build-hip\build.ninja -Pattern "mmq-instance-q4_k" -Context 0,8 | Select-Object -First 1
```

2 つ目の出力の `FLAGS =` の行に `-O3` があれば大丈夫です。

### 4.5 Linux との違い

- **iGPU レーンはありません。** RAM 段のエキスパートは CPU が計算します。Ryzen AI Max に単体のカードをつないだ場合も同じです。
- **GPU の番号** は ROCm の `hipInfo` の番号です。`"gpu_pci"` は使いません。カードを足したり外したりしたら、セットアップをもう一度
  実行してください。`"gpu"` にはカードの UUID も書けます（`build-hip\abyss.exe --list-gpus` で一覧が出ます）。ただし Windows の
  AMD ドライバが UUID を返すかはまだ確かめていません。返さないカードは一覧で `-` と出るので、番号で指定してください。
- **`--bench` の `PCIe ?`。** Windows では AMD のカードの PCIe のリンクを読めません（#43、#47、#49）。ベンチのほかの部分は使えます。
- **`HSA_OVERRIDE_GFX_VERSION`** は Linux の設定で、Windows では何もしません。
- **Ryzen AI Max だけで動かす場合:** Windows は iGPU にメモリを決まった量だけ割り当てます。その量が Windows 側の RAM より少ないと
  セットアップが警告し、「Variable Graphics Memory」（AMD Software の パフォーマンス > チューニング、または BIOS）を増やすよう案内します。
  たとえば 128 GB の PC なら 96 GB です。Windows での **報告なし**。

AMD について、Linux も含めた詳しい説明: [docs/amd](amd/README.ja.md)。

## 5. PC に合うモデル

`--quant` を付けないと、セットアップが PC に合うサイズを選び、`this PC: 16 GB GPU + 64 GB RAM -> --quant 16G64G (the others: ...)`
と表示して確認します。主なモデルは次のとおりです（AMD は `--backend hip` を足します）。

| GPU + RAM | モデル | コマンド | 書かれる config |
| --- | --- | --- | --- |
| 12 GB + 32 GB | Qwen3.8-Flash-Next 12G32G | `.\setup-deepseek.bat --model qwen38 --quant 12G32G --download-model` | `abyss-qwen38-12g32g.json` |
| 16 GB + 64 GB | Qwen3.8-Flash-Next 16G64G | `.\setup-deepseek.bat --model qwen38 --download-model` | `abyss-qwen38-16g64g.json` |
| 16 GB + 64 GB | Ornith-1.5-122B IQ3_XXS-B | `.\setup-deepseek.bat --model ornith --download-model` | `abyss-ornith-iq3_xxs-b.json` |
| 8〜16 GB + 32 GB | Ornith-1.5-35B-A3B Q4_K_M | `.\setup-deepseek.bat --model ornith35 --download-model` | `abyss-ornith35-q4_k_m.json` |
| 10〜12 GB、全部 VRAM | SenseNova-Loop-9B Q4_K_M | `.\setup-deepseek.bat --model sensenova-loop-9b --download-model` | `abyss-sensenova-loop-9b-q4_k_m.json` |
| 24 GB + 64 GB | Qwen3.8-Flash-Next 24G64G | `.\setup-deepseek.bat --model qwen38 --quant 24G64G --download-model` | `abyss-qwen38-24g64g.json` |
| 32 GB + 96 GB | GLM-5.3-Flash 32G96G | `.\setup-deepseek.bat --model glm --download-model` | `abyss-glm-32g96g.json` |
| VRAM + RAM で約 100 GB | DeepSeek V4 Flash 0731 UD-Q2_K_XL（`--model` なしの既定） | `.\setup-deepseek.bat --download-model` | `abyss-dsv4-q2_k_xl.json` |

- セットアップの最後の行に config の名前が出ます（`Done. Start it with: ...\abyss.sh start <名前>` の `<名前>` に対して、config は
  Abyss のフォルダの `abyss-<名前>.json`）。表と違う名前なら、そちらを使ってください。
- すべてのサイズとモデル、速さと品質: [README](../README.ja.md) の「PC に合うモデル」の表と [docs/MODELS](MODELS.ja.md)。
  8〜12 GB のカード: [docs/SMALL-VRAM](SMALL-VRAM.ja.md)。
- Windows の利用者が報告した数字（参考。`--bench` の生成の速さ）: RX 9060 XT 16 GB + 64 GB で Ornith-122B が 16〜17 tok/s（#43）、
  Qwen3.8 12G32G が 28 tok/s（#47）、Qwen3.8 16G64G が 22 tok/s（#49。v1.3.189 では 23.1 tok/s、2K のプロンプトの読み込みは 9 → 391 tok/s）、
  SenseNova-Loop-9B が 31.7 tok/s で 8K のプロンプトが 1,014 tok/s（#49、v1.3.189、Adrenalin 26.11.2）。RTX 4090 + 96 GB で DeepSeek V4 0731 が DSpark の下書きモデル
  なしで 37〜39 tok/s（#46）。

## 6. 起動と計測

### 起動

```powershell
cd C:\AI\Abyss
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json
```

起動したら http://127.0.0.1:8082/ を開きます（ダッシュボードと、OpenAI / Anthropic 形式の API）。止めるときは、そのウィンドウで
Ctrl+C を押します。

- **`--config` を付けてください。** セットアップの最後に `...\Abyss-tools\abyss.sh start <名前>` と出ますが、`abyss.sh` は bash の
  スクリプトなので PowerShell では動きません。`START-ABYSS.bat --config abyss-<名前>.json` を使います（#43）。v1.3.139 からは、
  `--config` なしの `START-ABYSS.bat` でも `setup-deepseek.bat` が書いた config を選べます（それより前は Maya の GLM のセットアップが
  始まっていました）。
- config は Abyss のフォルダの `abyss-<名前>.json`、エンジンのログはその隣の `abyss-<名前>.log` です。
  config の各項目の意味: [docs/SETTINGS](SETTINGS.ja.md)。

### 計測と調整（Abyss を止めてから）

```powershell
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --bench          # 数分: 速さ、何が律速か、品質の確認
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --bench --deep   # + PCIe と CPU レーンの分担の掃引
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --calibrate --no-start   # CPU レーンと PCIe の分担を調整、10〜15 分
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --check          # 安定性の確認、15〜30 分
.\START-ABYSS.bat --gpu-tune                                           # GPU ごとのカーネルの選び方、1 分、モデル不要
.\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --report         # abyss-report.txt を書く
```

- `--bench` は `abyss-bench.txt`、`--report` は `abyss-report.txt`、`--gpu-tune` は `abyss-gpu-tune.txt` を、どれも Abyss の
  フォルダに書きます。どこにも送信しません。ベンチの読み方: [docs/BENCH](BENCH.ja.md)。
- `--calibrate` は RTX 4090 で +12% でした（#39: 50.2 → 56.5 tok/s）。セットアップのあとに 1 回、PC の構成を変えたらもう 1 回
  実行してください。
- コーディングエージェントで使うなら、温度 0 で下書きモデルの採用率が上がりました（#39、RTX 4090）。マニュアルのコーディングツールの節
  （[MANUAL](MANUAL.ja.md)）を見てください。

### 1 回だけ設定を変える（PowerShell）

`$env:名前="値"` で、そのウィンドウを閉じるまで環境変数が設定され、そこから起動したエンジンにも渡ります。`Remove-Item` で元に戻します
（#49 で使った書き方）:

```powershell
$env:STRATA_GLM_TIMING="1"; $env:STRATA_GLM_PREFILL_PROF="1"; .\START-ABYSS.bat --config .\abyss-qwen38-16g64g.json --bench
Remove-Item Env:STRATA_GLM_TIMING, Env:STRATA_GLM_PREFILL_PROF -ErrorAction SilentlyContinue
```

ずっと使う設定は、config の `"env"` に書きます（`notepad .\abyss-qwen38-16g64g.json` で開き、値は `"STRATA_GLM_RAM_HEADROOM_GB": "6"`
のように引用符で囲んだ文字列にします）。書いたら Abyss を起動し直します。

## 7. RAM とページファイルと Windows の設定

### コミットの上限（RAM + ページファイル）

Windows は、固定（pinned）の RAM と、WDDM の下では GPU のメモリの確保も、コミットの上限で数えます。上限の方が先に尽きると、
エンジンのログに次のような行が出ます。

```
glm fast: WARNING - Windows' commit limit (RAM + page file) caps the RAM tier 9 GB below the free RAM (...): those
experts are read from the SSD. Raise the page file's maximum size by at least 13 GB (System > Advanced system
settings > Performance > Virtual memory) and restart Abyss
```

（v1.3.174 からの文面です。#43 では、RAM 64 GB の PC で RAM 段が 22.8 GB で止まり、この警告が出ていました。）

1. コミットの上限（`TotalVirtualMemorySize`、単位 KB）とページファイルを確かめます。

   ```powershell
   Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, TotalVirtualMemorySize, FreeVirtualMemory
   Get-CimInstance Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage
   ```

2. ページファイルの設定を開きます。

   ```powershell
   SystemPropertiesPerformance.exe
   ```

   「詳細設定」タブ >「仮想メモリ」の「変更」。「すべてのドライブのページング ファイルのサイズを自動的に管理する」のチェックを外し、
   ドライブを選んで「カスタム サイズ」に MB で入力し、「設定」、「OK」の順に押して、Windows を再起動します。
3. 大きさは、警告が求める量以上にします。報告では **64 GB**（65536 MB）でした。#4 は RAM 128 GB で固定 64 GB、#43 は RAM 64 GB で
   初期 24 GB / 最大 64 GB にして、RAM 段が 22.8 GB から 33.9 GB に増えました。ドライブにはその分の空きが要ります。
4. v1.3.174 からは、ページファイルがまだ伸びられる分（システム管理、または初期サイズ < 最大サイズ）も数えます。
   `STRATA_PAGEFILE_GROWTH=0` にすると、いまの大きさだけを数えます。
5. それでもログの RAM 段が RAM の割に小さければ、config の `"env"` で余白を減らします。#43 は `"STRATA_GLM_RAM_HEADROOM_GB": "6"`、
   #4 は `"4"` でした（AMD のセットアップは 16 を書きます）。Windows とほかのアプリの分は残してください。

RX 7900 XT + 128 GB で DeepSeek V4 のエキスパートを全部 VRAM と RAM に載せた設定（#4。AMD のセットアップが表示するもの）:
ページファイル 64 GB、`STRATA_GLM_RAM_HEADROOM_GB=4`、`STRATA_GLM_RESERVE_MB=512`（そのカードに画面をつないでいないときだけ）、
`--prefill 8192`。

### NVIDIA: Sysmem Fallback を切る

NVIDIA コントロールパネルの「3D 設定の管理」で「CUDA - システム メモリ フォールバック ポリシー」(CUDA - Sysmem Fallback Policy)
を「システム メモリ フォールバックを優先しない」(Prefer No Sysmem Fallback) にします。既定のままだと、カードが満杯になったとき
(ブラウザやデスクトップが VRAM を使った後など) ドライバーがエラーなしで VRAM の確保をホスト RAM に回し、decode が大きく遅くなります。
Abyss も同じ理由で、Windows では VRAM を 1.5〜2 GB 空けておきます (v1.3.195 以降)。

### RAM より先に固定メモリが尽きる場合

GPU のドライバが、RAM が埋まる前に固定メモリの確保を断ることがあります。

- NVIDIA（#31、RTX 4090、`STRATA_GLM_RAM_GB=72`）: 約 2 万トークン以上のプロンプトが失敗し、約 4 万トークンでエンジンが落ちていました。
  v1.3.107 からは、プロンプトの CPU の分担を PCIe に回し、`the CPU lane's pinned host buffers did not allocate` と表示します。
  この PC では `"STRATA_GLM_PREFILL_CPU": "0"` の方がわずかに速くもありました（#31、#46）。
- AMD（#49、RX 9060 XT）: 起動が `pinned disk staging did not allocate` で止まっていました。v1.3.181 からは、RAM 段の枠を返して起動を
  続けます（`gave back ... GB of RAM-tier slots`）。

### P コアと E コアのある Intel CPU

config の `"env"` に `STRATA_GLM_CPU_PIN=0-15` を書くと、CPU レーンのワーカーを P コアのスレッドに固定します（i7-14700K の P コアの
スレッドは 16 本で、0〜15。#4、#12）。v1.3.56 からは、そのワーカーは Windows の省電力の抑制（EcoQoS）も受けません。
タスクマネージャーの「関係の設定」で Abyss のプロセス全体を P コアに **縛らないでください**。GPU のランタイムのスレッドが動けなくなり、
37 → 19 tok/s、ひどいときはさらに遅くなりました（#9、#12）。

### 24 GB のカードと DSpark の下書きモデル

RTX 4090 では、DeepSeek V4 は DSpark の下書きモデルなしの方が速く動きました（なしで 37〜39 tok/s、ありで 33〜38）。下書きモデルが使う
6.5 GB の VRAM を、エキスパートの枠に回した方が効くためです（#46）。エンジンが DSpark を自動で使うのは、48 GB 以上のカードだけです。

## 8. 更新

- ダッシュボードの About > Update。git で取ったフォルダで、手を加えたファイルが無いときに使えます。
- 手で行う場合:

  ```powershell
  cd C:\AI\Abyss
  git pull
  .\setup-deepseek.bat --backend hip --model qwen38      # AMD。NVIDIA は --backend hip なしで同じ
  ```

  セットアップは、変わったところだけビルドし直すか（AMD）、新しい版のビルド済みエンジンを取ってきます（NVIDIA）。モデル、パック、
  config はそのまま残します（`--rewrite-config` を付けると config を書き直します。ポート、ホスト、API キー、MCP サーバーは残ります）。
  起動（`START-ABYSS.bat --config ...`）と、v1.3.139 からは `--bench` も、変わったエンジンを先にビルドし直します。

## 9. トラブル対応

| 症状 | 原因 | 対処 | issue |
| --- | --- | --- | --- |
| `64-bit Python 3.10 or newer is needed` と出る | Python が無い、または Microsoft Store のダミーしか無い | `winget install -e --id Python.Python.3.12 --scope user` のあと、ウィンドウを開き直して `.bat` をもう一度 | - |
| PowerShell で `.\setup-deepseek.bat` が「認識されません」 | Abyss のフォルダにいない、または `.\` が無い | `cd C:\AI\Abyss` のあと `.\setup-deepseek.bat` | - |
| NVIDIA: セットアップがドライバのところで止まる | ドライバが 580 より古い（ビルド済みエンジンは CUDA 13.3） | https://www.nvidia.com/drivers から更新（610 以上を推奨） | - |
| NVIDIA で自分でビルド: `ptxas error : Instruction 'mma with block scale' not supported on .target 'sm_120'` | RTX 50 向けに `120` で手ビルドした | `120a` でビルドする（セットアップはそうする） | #40 |
| AMD: `Visual Studio Build Tools (2022 or 2026) are needed` | C++ のワークロードが無い | [4.1](#41-インストール) の winget のコマンド | - |
| AMD: `requires HIP 7.14 or newer ... found HIP 7.x` | システムの HIP SDK（ROCm 7.2）か `ROCM_PATH` が先に見つかった | `.venv` に SDK を手で入れる: [4.3](#43-rocm-sdk-を手で入れる) | #43 |
| AMD: 初回に `rocm-sdk init` で数分止まって見える | 初回は約 3 GB を展開する | 待つ。2 回目からは確認だけ | #43、#49 |
| AMD: `ROCm's hipInfo sees no AMD GPU` | ドライバが古いか、入っていない | Adrenalin を更新して再起動 | - |
| AMD: CMake の HIP コンパイラのテストが `__device__ function 'isgreater' cannot overload __host__ __device__ function 'isgreater'` で止まる | Visual Studio 2026（MSVC 14.51）と ROCm SDK 7.14.1 の Clang 23 のヘッダの衝突 | v1.3.131 以降はセットアップが SDK のヘッダを自動で直す。`ABYSS_NO_ROCM_HEADER_FIX` が設定されていないか確認。セットアップがファイルを書けないときは、`.venv\Lib\site-packages\_rocm_sdk_devel\lib\llvm\lib\clang\23\include\__clang_hip_runtime_wrapper.h` の `#include <__clang_cuda_math_forward_declares.h>` を、最初の `#include <cmath>` のすぐ上に移す | #2、#43 |
| AMD: ビルドが `glm_prefill.cu` の `float* OUT = M.OUTP;` で `error: expected unqualified-id` になって止まる | `<windows.h>` が `OUT` を空のマクロとして定義していて、v1.3.186 で入った HIP 専用の関数が変数に `OUT` という名前を使っていた（v1.3.186〜v1.3.188） | v1.3.189 で直った。更新してセットアップをやり直す | #49 |
| AMD: ビルド中に警告が何百も出る（`-Wsign-compare`、`-Wunused-function` など） | 正常 | 何もしなくてよい | #49 |
| AMD: 生成は普通なのに、プロンプトの読み込みが数 tok/s（SenseNova 3、Qwen3.8 9 tok/s） | v1.3.188 より前のエンジンで、MMQ のカーネルが `-O0` でビルドされていた | 更新してセットアップをもう一度。[4.4](#44-ビルドが最適化されているか確かめる) の方法で確かめる | #3、#43、#49 |
| AMD: 何もかも極端に遅い（1 tok/s 前後） | 全カーネルが `-O0`（v1.3.29 より前） | 更新 | #3 |
| AMD: `--bench` が `the engine stopped during the benchmark` で止まり、ログに `amdhip64_7.dll` の中での停止（`hipblasGemmEx` → `hipblasLtMatmul`）がある | ROCm SDK 7.14.1 の hipBLASLt | v1.3.182 以降（自前の行列積）と v1.3.188 以降（`ROCBLAS_USE_HIPBLASLT=0` が既定）。古い版ではコマンドの前に `$env:ROCBLAS_USE_HIPBLASLT="0"` | #49 |
| AMD: 起動が `pinned disk staging did not allocate` で止まる | RAM より先に、ドライバが固定できるメモリが尽きた | v1.3.181 以降は RAM 段の枠を返して続ける | #49 |
| AMD: `--bench`、`--check`、セットアップの画面に古いカードの名前が出る（RX 9060 XT 16 GB なのに「RX 6600 XT 8 GB」） | Windows のアダプタの登録情報に、外したカードが残っている | v1.3.139 以降は ROCm の hipInfo に聞く。表示だけの問題で、エンジンは VRAM を自分で測っている | #43 |
| AMD: ベンチに `PCIe ?` と出る | Windows では AMD のカードのリンクを読めない | 何もしなくてよい（ほかの結果は有効） | #43 |
| `--config` なしの `START-ABYSS.bat` が Maya の GLM-5.3 のセットアップを始める | v1.3.139 より前は Maya の config だけを探していた | 更新するか、`--config .\abyss-<名前>.json` を付ける | #43 |
| 更新後の `--bench` が古いエンジンを測っている | v1.3.139 より前はビルドし直さなかった | 更新する。または `git pull` のあとにセットアップを 1 回実行 | #43 |
| ログの RAM 段が RAM よりずっと小さく、`glm stat` に `disk` の読み込みが多い | コミットの上限（ページファイル） | [7 章](#7-ram-とページファイルと-windows-の設定) | #4、#43 |
| NVIDIA: 大きいプロンプト（約 2 万トークン以上）が `host buffers did not allocate` で失敗する、またはエンジンが `0xC0000005` で終わる | CPU レーンの固定バッファ（v1.3.107 より前） | 更新。`"STRATA_GLM_PREFILL_CPU": "0"` も選べる | #31 |
| DeepSeek V4.1: `an invalid Engram row (NaN code or scale)` | Windows でファイルの位置が 32 ビットだった（v1.3.34 より前） | 更新 | #6 |
| タスクマネージャーで Abyss を P コアに縛ったら全体が遅い | GPU のランタイムのスレッドが動けない | 関係の設定を外し、`STRATA_GLM_CPU_PIN` を使う（[7 章](#p-コアと-e-コアのある-intel-cpu)） | #4、#9、#12 |
| 古いビルドエラー: `cudaMemcpyKind`、`M_PI`、`sys/socket.h`、`__builtin_memcpy`、`_cvtsh_ss` | 修正済み（v1.3.29、v1.3.141、v1.3.120、v1.3.130、v1.3.159） | 更新 | #2、#31、#39 |

## 10. 報告のしかた

このページは Windows の PC からの報告でできています。結果も問題も、ぜひ送ってください。

- **速さ:** `.\START-ABYSS.bat --config .\abyss-<名前>.json --bench`（または `--bench --deep`）を実行し、`abyss-bench.txt` を
  [速度の報告](https://github.com/zephel01/Abyss/issues/new?template=bench.md) に貼ります。
- **問題:** `.\START-ABYSS.bat --config .\abyss-<名前>.json --report` を実行し、
  [不具合の報告](https://github.com/zephel01/Abyss/issues/new?template=bug.md) に `abyss-report.txt` とコンソールの出力を貼ります。
  ビルドエラーなら、コンパイラの最初の `error:` の行も入れてください。
- **こちらに無い GPU:** `abyss-gpu-tune.txt` を [GPU tune の報告](https://github.com/zephel01/Abyss/issues/new?template=gpu-tune.md)
  に貼ります。
- 書いてほしいこと: Windows 11、GPU とドライバの版、RAM とページファイルの大きさ、Abyss の版、NVIDIA ならビルド済みか自分でビルドか
  （と CUDA の版）、AMD なら ROCm SDK の版と Visual Studio の版。

ファイルにホスト名、ユーザー名、フォルダのパスは入りません。貼る前に中身を読んで確かめてください。

## 11. Windows で確かめられていないこと

- Windows での NVIDIA のビルド済みエンジン（CI でビルドと確認はしているが、利用者の報告はまだ無い）。
- NVIDIA で自分でビルドするときに「x64 Native Tools Command Prompt」が要るかどうか（報告者はどう実行したかを書いていない）。
- Windows での AMD の 2 枚構成、Ryzen AI Max の iGPU だけでの実行、RX 9070 / R9700、RX 7000 の同系統のカード。
- ウイルス対策（Microsoft Defender）や SmartScreen の警告、Windows のパスの長さの制限: 問題の報告はまだありません。
