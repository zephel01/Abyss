# Abyss マニュアル

Abyss のビルド、DeepSeek V4 Flash（と V4.1 Flash）の準備、起動と調整の手順。English: [MANUAL.md](MANUAL.md)

Abyss は [Project Maya](https://github.com/mw00/project-maya)（MIT）をコピーして作ったフォークで、Maya は
[Strata](https://github.com/Niko1221/Strata)（MIT）から生まれたプロジェクト。エンジン、エキスパートの配置の仕組み、
サーバー、ダッシュボードは Maya のもので、Abyss はそこに DeepSeek V4 と V4.1 への対応を足した。Maya の元のドキュメント
（GLM-5.3-Flash、AMD、複数 GPU、インストーラ）は [docs/maya/](maya/README.md) にそのまま残している。

- [必要なもの](#必要なもの)
- [まとめてセットアップ](#まとめてセットアップ)
- [ビルド](#ビルド)
- [モデルを取る](#モデルを取る)
- [pack を作る](#pack-を作る)
- [設定と起動](#設定と起動)
- [使い方](#使い方)
- [DeepSeek V4.1 Flash](#deepseek-v41-flash)
- [Qwen3.8-Flash-Next](#qwen38-flash-next)
- [Ornith-1.5-122B](#ornith-15-122b)
- [EXL3 量子化（GLM-5.3-Flash）](#exl3-量子化glm-53-flash)
- [メモリと速度](#メモリと速度)
- [DSpark のドラフトモデル（DeepSeek V4 Flash）](#dspark-のドラフトモデルdeepseek-v4-flash)
- [実行時の無検閲化（`--uncensor`）](#実行時の無検閲化--uncensor)
- [設定項目](#設定項目)
- [llama.cpp との照合](#llamacpp-との照合)
- [うまくいかないとき](#うまくいかないとき)
- [制限](#制限)

## 必要なもの

| | 確認した環境 | 補足 |
| --- | --- | --- |
| OS | Ubuntu (Linux) | Windows 11 は利用者が確認（`setup-deepseek.bat`。RTX 4090 #31、RX 7900 XT #5）。こちらでは未確認 |
| GPU | RTX 5090 (32 GB)、RTX 3090 | GPU のアーキテクチャに合わせてビルドする（RTX 50 は `120a`、RTX 30 は `86`。分割するときは複数まとめて）。Blackwell より前の GPU では MXFP4 のエキスパートは FP4 MMA ではなく q8_1 の経路で計算する。Ampere（sm_80）より前では indexer と prompt の attention は従来の FP32 カーネルを使う。V100（sm_70）は CUDA 12 でビルドできるが実機は未確認。AMD Radeon は「AMD の GPU」（試験的） |
| RAM | 94 GB | モデルは 96.8 GB。VRAM + RAM が約 100 GB を超えれば全エキスパートがメモリに載る。足りない分は SSD から読む（動くが遅い） |
| ディスク | NVMe SSD、空き約 100 GB | GGUF ファイル。pack は 0.2 GB |
| CUDA | 13.3 | `nvcc` を含む。V100 は 12.9（`--nvcc`） |
| コンパイラ | g++ 13 | CUDA が受け付けるバージョン |
| Python | 3.12 | Abyss のフォルダ内の `.venv` |
| ROCm（任意） | 10（HIP 7.14 以上、`hipcc`） | AMD APU の内蔵 GPU が RAM 層のエキスパートを計算する（下の「iGPU レーン」）。なければ CPU が計算する |
| ARM（ベータ） | NVIDIA DGX Spark（GB10、aarch64） | エンジンは CUDA 12.9 以上で `121a` 向けに手でビルドする。下の「NVIDIA DGX Spark」 |

`./abyss.sh`（Maya のインストーラの名前を変えたもの）は、今も Maya の GLM-5.3-Flash を入れるためのもの。
DeepSeek V4 は `./setup-deepseek.sh`（次の節）か、その後の手順を手で行って準備する。

## まとめてセットアップ

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model
```

DeepSeek V4.1 Flash 1017 は、GPU 32 GB + RAM 96 GB 向けの zephel01 の LDLQ GGUF で入れられる（BETA。既定は速さ重視の
132G、品質重視は `--quant 139G`。[モデルカード](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF)）:

```sh
./setup-deepseek.sh --model v41 --download-model
```

小さくて汎用の [Compact JA256](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF)（207 GB、エキスパート 384 個中 256 個、日本語は崩れない、32 GB + 96 GB で約 3 割速い）は、まだセットアップに入っていない。`JA256/` フォルダをダウンロードし、[docs/deepseek41/README.md](deepseek41/README.md#ggufs-that-load) のとおり手作業で pack する。

設定 `abyss-dsv41-132g.json` には、モデルカードの普段使いの設定（`STRATA_GLM_TAIL_SKIP=4`、`STRATA_GLM_TAIL_SKIP_DISK=1`、
`STRATA_GLM_ROUTE_AFFINITY_RAM=0.08`、ディスク読み出し 8 本）が入る。V4.1 の DSpark ドラフターはセットアップしない
（手作業: [docs/deepseek41](deepseek41/README.md)）。ほかの V4.1 の GGUF も、そこに書いた手作業で準備する。

Qwen3.5-MoE の 122B モデル 2 つは、GPU 16 GB + RAM 64 GB 向けの zephel01 の量子化版で入れられる（51〜52 GB。Ornith は
2 つ目のシャードにある NextN 層で下書きする）:

```sh
./setup-deepseek.sh --model ornith --download-model     # Ornith-1.5-122B-IQ3_XXS-16G64G
./setup-deepseek.sh --model qwen35 --download-model     # Qwen3.5-122B-A10B-IQ3_XXS-16G64G
```

dense モデルの Qwen3.8-27B は unsloth の UD-Q4_K_XL（17.7 GB、v1.3.138）で入れられる。24 GB 以上の GPU はファイル全体を
VRAM に載せ、v1.3.140 から 12〜20 GB の GPU でも、入らない FFN の行をピン留めした RAM に置いて動かせる（RAM 32 GB が必要）。
v1.3.142 からは 8 GB の GPU でも動く（8K）。8 GB 未満の GPU では何もダウンロードする前に止まり、GPU に入る文脈の長さを
設定に書く（8〜11 GB: 8K、24 GB: 32K、32 GB: 128K。[Qwen3.8-27B](#qwen38-27bdense)）:

```sh
./setup-deepseek.sh --model qwen38-27b --download-model   # Qwen3.8-27B-UD-Q4_K_XL（17.7 GB）
./setup-deepseek.sh --model qwen38-27b --quant UD-Q6_K --download-model   # リポジトリのほかのファイル
```

`--quant` には [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) のどのファイルでも指定できる
（UD-IQ1_S〜UD-Q8_K_XL、Q4_0、Q4_1、Q8_0、BF16）。指定しなければ、UD-IQ2_XXS / UD-IQ3_XXS / UD-Q4_K_XL / UD-Q6_K / Q8_0
のうち GPU にすべて載る最大のものを示し、既定は UD-Q4_K_XL のまま。NextN ブロックのないファイル（UD-IQ1_S、UD-IQ1_M、
UD-IQ2_XXS、UD-IQ2_XS、UD-IQ2_S）では、unsloth の別ファイル（`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`、1.4 GB）も取ってくる。
そのあと画像を使えるようにするかを聞く（unsloth の `mmproj-F16.gguf` 0.93 GB と `strata-vision` のビルド。
`--no-vision` で飛ばす）。v1.3.171 から、ファイル・コンテキスト・下書き役が NVIDIA の GPU 1 枚の VRAM に収まるとき
（UD-Q4_K_XL なら 24 GB 以上）は、z-lab の DFlash 2 下書き役
（[z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) の Q4_K_M、1.14 GB。Hugging Face の
SHA-256 で確認）も取ってきて、設定に書く（`STRATA_Q35_DFLASH`）。`--no-dflash` で取らない（設定にあれば外す）。
`--dflash` で小さい GPU でも入れる（[下書き役](#qwen38-27bdense)）。

Ternary Bonsai 2.5 27B（prism-ml の 3 値の Qwen3.8-27B、ベータ）: PTQ1_0 のファイル（6.23 GB、MTP ブロックつき）を 8 GB の
GPU からすべて VRAM に載せる（8 GB: 8K、9.5 GB: 32K、12 GB: 64K、16 GB: 128K。[Ternary Bonsai](#ternary-bonsai-25-27bベータ)）:

```sh
./setup-deepseek.sh --model bonsai27b --download-model              # Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf（6.23 GB）
./setup-deepseek.sh --model bonsai27b --quant Q2_0 --download-model # Q2_0 のファイル（7.67 GB、同じ重み、MTP ブロックなし）
```

Inkling 2 Mini abliterated（orcarouter が abliterated 版を型の混在で量子化した GGUF。この版から）: `Mix-16G`（15.7 GB、既定）
か `Mix-12G`（11.6 GB）。どちらも 16〜24 GB の GPU で全部 VRAM に載る。Inkling 2 Mini と同じ `inkling2_loop` の形で、
llama.cpp の標準の型を層ごとに混ぜている（16G は gate-up が NVFP4 / IQ4_XS / Q5_K、down が IQ4_NL / Q5_1。12G は gate-up が
IQ2_XXS〜IQ3_S、down が IQ4_NL）。unsloth の Inkling 用 DSpark drafter がそのまま使える（使っても greedy の token は同じ。
md5 で確認）。`--model inkling2mini` が置く場所にそのファイルがあれば、エンジンが読み込む。
RTX 5090: decode 195 / 215 tok/s（16G / 12G。code で drafter ありなら 275 / 288）、7,217 トークンのプロンプト 7,265 / 7,787 tok/s。
RTX 3090（全部 VRAM）: decode 104 / 114 tok/s、同じプロンプト 1,700〜1,930 / 2,950 tok/s:

```sh
./setup-deepseek.sh --model inkling2mini-abl --download-model                  # Inkling-2-Mini-abliterated-Mix-16G（15.7 GB）
./setup-deepseek.sh --model inkling2mini-abl --quant Mix-12G --download-model  # Mix-12G（11.6 GB）
```

Ling-3.2-air（inclusionAI、`bailingmoe3`。この版からベータ。[数字](MODELS.ja.md#ling-32-airベータv13201)）:
bartowski が公開している Q4_K_M（3 分割、80.4 GB）をそのまま pack する。GPU 24〜32 GB + RAM 64〜96 GB 向け。
画像も読める（NVIDIA。セットアップが聞く。[画像](SETTINGS.ja.md#ling-32-air-の画像)）。MTP の下書き役はまだ無い。会話スロットは Qwen3.8 と同じ（`STRATA_GLM_SLOTS`）。設定にはモデルのサンプリング（temperature 1.0、top-p 0.95、
top-k 20）が入る。RTX 5090 + RAM 123 GB: 4K のプロンプト 2,061 tok/s、decode 142.8 tok/s。16K では 3,092 / 120.7。

サーバーでは、いちばん最初の起動直後の回答が遅めになる。チャット 8 回（英語 / 日本語 / コード、thinking あり、
200〜700 トークン）で decode 88〜116 tok/s（中央値 99）、TTFT 0.16〜0.38 秒。エンジンは答えながらこのマシンでの
エキスパートの使われ方を覚え（pack の `expert_usage.txt`）、次の起動でそれに沿って VRAM を埋める。その約 4K トークンの
あとの起動では 124〜141 tok/s（中央値 135）、TTFT 0.13〜0.32 秒。このモデルでは VRAM の置き場が会話の途中では
動かないので、速くなるのは回答を重ねたときではなく次の起動から（`STRATA_GLM_BG_SPLIT=1` でも数字は同じ）。
配布する経路の履歴（`STRATA_GLM_PRIOR`、Qwen3.8 にあるもの）があれば初回から速くなるが、Ling 用はまだ無い。

確かめたのは Q4_K_M だけ。`--quant` には bartowski の他のファイル（IQ1_S〜Q8_0、bf16）を名前で渡せるが、未確認:

```sh
./setup-deepseek.sh --model ling32air --download-model                 # Ling-3.2-air-Q4_K_M（80.4 GB）
./setup-deepseek.sh --model ling32air --quant IQ4_XS --download-model  # repo の他のファイル（未確認）
```

Ling-3.2-air の画像: セットアップは bartowski の `mmproj-Ling-3.2-air-f16.gguf`（0.87 GB）も入れるかを聞き、
`abyss-vision`（このモデル用に Abyss が自前で書いたエンコーダー。エンジンと一緒にコンパイルされ、llama.cpp は使わない）で動かす。
OpenAI の `image_url`、Anthropic の `image` ブロック（`tool_result` の中も）がモデルに届く。RTX 5090 で、画像トークン
240〜375 の画像のエンコードは 23〜42 ms。画像 1 枚つきの回答全体（プロンプト 274〜411 トークン、出力 36〜67 トークン）は
1.1〜2.4 秒だった。文字だけの要求は、画像なしのときと同じトークンになる。

QSA indexer 付きの小さい dense モデル Qwen3.5-9B QSA（zephel01 の BETA。[Qwen3.5-9B QSA](#qwen35-9b-qsa)）は、ファイル全体を
11 GB 以上の GPU 1 枚に載せる。その GPU で 128K までの文脈が動く（VRAM を 11 GB に制限して計測）。それより小さい GPU では
何もダウンロードする前に止まる。既定は Q4_K_M（5.9 GB）、もう 1 つは `--quant Q8_0`（9.9 GB。メモリは未計測で、
セットアップは差の 4 GB を足して数える）。どちらのファイルにも NextN の下書き層が入っている。画像は使えない
（画像の変換器のファイルがない）。`--model ornith-9b-qsa` は同じものの Ornith-1.5-9B 版（zephel01/Ornith-1.5-9B-QSA-GGUF、[Ornith-1.5-9B QSA](#ornith-15-9b-qsa)）で、GPU の表も同じです:

```sh
./setup-deepseek.sh --model qwen35-9b-qsa --download-model                    # Qwen3.5-9B-QSA-Q4_K_M（5.9 GB）、文脈 64K
./setup-deepseek.sh --model qwen35-9b-qsa --context 131072 --download-model   # 同じファイルで文脈 128K
./setup-deepseek.sh --model ornith-9b-qsa --download-model                    # Ornith-1.5-9B-QSA-Q4_K_M（5.9 GB）、文脈 64K
./setup-deepseek.sh --model ornith-9b-qsa --context 131072 --download-model   # 同じファイルで文脈 128K
./setup-deepseek.sh --model mimo-9b --download-model                          # MiMo-V2.6-Distill-Qwen-9B-Q4_K_M（5.8 GB）、文脈 64K
./setup-deepseek.sh --model mimo-9b-qsa --download-model                      # MiMo-V2.6-Distill-Qwen-9B-QSA-Q4_K_M（5.9 GB）、文脈 64K
./setup-deepseek.sh --model sensenova-loop-9b --download-model                # SenseNova-Loop-9B-Q4_K_M（5.6 GB）、文脈 64K
```

`--model mimo-9b`（v1.3.168）は Xiaomi の [MiMo-V2.6-Distill-Qwen-9B](#mimo-v26-distill-qwen-9b) を bartowski の GGUF から入れます
（既定は Q4_K_M 5.8 GB、`--quant Q8_0` で 9.5 GB）。GPU の表は Qwen3.5-9B QSA と同じです（形も最大値も同じ。MiMo では VRAM を制限した
計測はしていません）。ファイルに NextN ブロックがないので、取れるときは Qwen3.5-9B のもの（0.16 GB）を GGUF の横の `MTP/` に
取ってきます。ないときは n-gram の下書きだけで動きます（生成が 20〜25% ほど遅い）。

`--model mimo-9b-qsa`（v1.3.170、BETA）は、同じものに Qwen3.5-9B QSA の indexer を移植したものです（zephel01/MiMo-V2.6-Distill-Qwen-9B-QSA-GGUF、
Q4_K_M 5.9 GB、`--quant Q8_0` で 9.6 GB。[MiMo-V2.6-Distill-Qwen-9B QSA](#mimo-v26-distill-qwen-9b-qsa)）。下書きブロックも同じく借りたものです。
GPU の表は制限して計測しました: 文脈 32K が 11 GB、128K が 12 GB（Q8_0 は 3.7 GB 多く）。

`--model sensenova-loop-9b`（v1.3.169）は [SenseNova-Loop-9B](#sensenova-loop-9b) を guideboardlabs の GGUF から入れます（既定は Q4_K_M 5.6 GB、
`--quant Q3_K_M` で 4.6 GB、`Q4_0` で 5.3 GB、`Q8_0` で 9.5 GB）。各ファイルは Hugging Face の SHA-256 で確かめます。GPU の表は
このモデル自身のもの（制限して計測）: 文脈 32K で 10 GB、64K で 12 GB、128K で 14 GB。GPU 1 枚で動きます。下書きブロックはなく、n-gram の下書きを使います。

zephel01 のほかの GGUF は PC の大きさごとに 1 つのリポジトリで、`--quant` で大きさを選ぶ（設定の RAM の枠は、この PC の RAM
から、モデルカードを測ったときの値にする。Qwen3.8 はシャードにある MTP 層で下書きする。V4 の自作版は単体 GPU では DSpark
なしのほうが速いので、DSpark は入れない）:

```sh
./setup-deepseek.sh --model ornith --quant 24G64G --download-model   # Ornith-1.5-122B-IQ4_XS-24G64G（68.9 GB）
./setup-deepseek.sh --model qwen38 --quant 12G32G --download-model   # Qwen3.8-Flash-Next-E448-IQ2_M-12G32G（67.5 GB）
./setup-deepseek.sh --model qwen38 --quant 16G64G --download-model   # Qwen3.8-Flash-Next-IQ3_M-16G64G（86.9 GB）
./setup-deepseek.sh --model qwen38 --quant 24G64G --download-model   # Qwen3.8-Flash-Next-IQ3_M-24G64G（93.7 GB）
./setup-deepseek.sh --model glm --quant 32G96G --download-model      # GLM-5.3-Flash-IQ2_XS-32G96G（94.1 GB）
./setup-deepseek.sh --model glm --quant 32G128G --download-model     # GLM-5.3-Flash-IQ3_XXS-32G128G（127.9 GB）
./setup-deepseek.sh --quant 16G64G --download-model                  # DeepSeek-V4-Flash-0731-IQ1_S-16G64G（61.7 GB）
./setup-deepseek.sh --quant 32G64G --download-model                  # DeepSeek-V4-Flash-0731-IQ2_XXS-32G64G（72.0 GB）
./setup-deepseek.sh --quant 32G96G --download-model                  # DeepSeek-V4-Flash-0731-IQ2_M-32G96G（96.8 GB）
```

`--quant` を付けないと、この PC に合う大きさを選ぶ。GPU の大きさがこの GPU の VRAM に、RAM の大きさがこの PC の RAM に収まる
いちばん大きいファイルを、ほかの候補と並べて表示し（`this PC: 16 GB GPU + 62 GB RAM -> --quant 16G64G (the others: 12G32G,
24G64G)`）、それを既定にして確認する（`--yes` ならそのまま）。Ryzen AI Max のような APU は共有メモリを 1 回だけ、RAM として数える。
どれも収まらないときは最小のファイルにして、その理由（SSD から読む分が増えて遅い）を出す。Ornith は 122B のファイルから選び、
397B も収まるときは注記で知らせる。前に入れた大きさ（その `abyss-<名前>.json` がある）はそのまま使うので、セットアップを
もう一度実行しても別のダウンロードは始まらない。DeepSeek V4 は既定の unsloth `UD-Q2_K_XL` のままで、この PC に合う
zephel01 のファイルを知らせるだけ。`--quant` を付ければいつもそれが優先。

`setup-deepseek.sh` は `.venv` を作ってから `tools/abyss_setup.py` を実行する。済んだ手順は飛ばすので、何度実行してもよい:

1. PC の確認（NVIDIA GPU、RAM）
2. Python パッケージ、固定コミットの llama.cpp のソース
3. GPU に合わせてエンジンをビルド（RTX 50 は `120a`）
4. モデルを `../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<量子化>` にダウンロード（`--download-model` が
   なければ先に確認する）。全ファイルを Hugging Face の SHA-256 と照合する
5. シャードと同じフォルダに pack を作る
6. Abyss のフォルダに設定 `abyss-<名前>.json` と起動スクリプト `run-abyss-<名前>.sh` を書く
7. DSpark のドラフトモデル（v1.3.43、任意）: 設定が新しいときは、DeepSeek の DSpark ドラフトモデル（6.9 GB）をダウンロードして
   使うかを確認する（`--dspark` なら確認なし）。SHA-256 を照合し、設定の `env` に書く
   （[DSpark のドラフトモデル](#dspark-のドラフトモデルdeepseek-v4-flash)）
8. Abyss の隣に **Abyss-tools** フォルダを作る（`abyss.sh start | stop | restart | status | update | build | log | models`。
   雛形は本体の `ops/abyss-tools/`）。セットアップを再実行すると `abyss.sh` / `common.sh` は雛形に合わせて更新し
   （前のものは `.bak` に残す）、自分で書いた `README.md` はそのまま残す

```sh
../Abyss-tools/abyss.sh start      # 起動後 http://127.0.0.1:8082/
../Abyss-tools/abyss.sh stop
```

| オプション | |
| --- | --- |
| `--quant UD-IQ3_XXS` | unsloth/DeepSeek-V4-Flash-0731-GGUF のほかの量子化（既定 `UD-Q2_K_XL`。確認済みはこれだけ） |
| `--name 名前` | `abyss-<名前>.json` の名前（既定 `dsv4-<量子化>`。例 `dsv4-q2_k_xl`） |
| `--context N`, `--port N` | コンテキスト長（既定 65536）とポート（既定 8082） |
| `--data-dir DIR`, `--tools-dir DIR` | Abyss-data と Abyss-tools を別の場所にする |
| `--gpu N`, `--arch A` | 使う GPU、コンパイルする CUDA アーキテクチャ |
| `--nvcc PATH` | ビルドに使う CUDA コンパイラ（V100 には CUDA 12 のもの） |
| `--gpus 0,1`, `--gpus all` | 複数の GPU に層を分けて載せる（下の「複数の GPU」）。全 GPU のアーキテクチャをまとめてビルドする |
| `--download-model`, `--yes` | 確認せずにダウンロード、すべての質問に既定値で答える |
| `--rewrite-config` | `abyss-<名前>.json` を書き直す（なければ既存のものを残す。書いてあった DSpark ドラフトモデルはそのまま使い、ポート、ホスト、API キー、MCP サーバーなどサーバーの設定も残す） |
| `--dspark`, `--no-dspark` | DeepSeek の DSpark ドラフトモデル（6.9 GB、`../Abyss-data/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF` へ）をダウンロードして設定でオンにする / 設定でオフにする（ファイルは残す）。GPU 1 枚のときだけ。`--model inkling2mini` では drafter（unsloth の `dspark-Inkling-2-Mini-Q8_0_Old.gguf`。`STRATA_INK_DSPARK` を参照）はダウンロードしない。ファイルがあれば、expert が横に収まるときエンジンが読み込む（`STRATA_INK_DSPARK`）。`--no-dspark` は `STRATA_INK_DSPARK=0` を書く |
| `--tools-only` | Abyss-tools だけを作る / 更新する |
| `--prebuilt` | エンジンをコンパイルせず、ビルド済みのものをダウンロードする（`nvcc` か C++ コンパイラが無いときは自動でこちらになる。下を参照） |


### コンパイラ無しで: ビルド済みエンジン（試行中）

CUDA ツールキット（`nvcc`）か C++ コンパイラ（Windows は Visual Studio Build Tools、Linux は g++）が無いとき、
セットアップ（`./setup-deepseek.sh`、`setup-deepseek.bat`、`./abyss.sh`、`START-ABYSS.bat`）はエンジンをコンパイルせず、
GitHub Actions でビルドしたものをダウンロードする。ツールキットがあっても `--prebuilt` でこちらを選べる。
ツールキットがあれば、これまでどおりコンパイルする。

| | |
| --- | --- |
| アーカイブ | `abyss-engine-cuda-linux-x64.tar.gz`、`abyss-engine-cuda-windows-x64.zip`。このバージョンの GitHub リリース（無ければ最新のリリース）から取り、リリースの `SHA256SUMS` で確かめて `build/` に展開する |
| GPU | RTX 30（sm_86）、RTX 40（sm_89）、RTX 50（sm_120a）を 1 本のバイナリで。ほかの GPU（V100、RTX 20、A100、H100、DGX Spark）はこれまでどおりコンパイルする |
| NVIDIA ドライバ | **580 以上が必須**（CUDA の minor version compatibility で CUDA 13.x が動く。セットアップは警告を出す。未検証）。**610 以上を推奨**（CUDA 13.3 のブランチ R610、Linux は 610.43.02）。580 未満ならセットアップは止まり、ドライバのページ https://www.nvidia.com/drivers を示す |
| 中身 | エンジンと `abyss_gpu_tune`、それが読み込む NVIDIA の CUDA ライブラリ（cuBLAS、cuBLASLt。Linux は CUDA ランタイムも）を手を加えずに同梱。[CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html#attachment-a) に従って配布 |
| Linux | Ubuntu 22.04 以降（glibc 2.35）。エンジンは横に置いたライブラリを読む |
| まだ無いもの | 画像エンコーダ（`strata-vision`）はコンパイルが要る。ビルド済みエンジンでは、モデルはテキストだけを読む |

`ABYSS_PREBUILT_URL` で、`SHA256SUMS` とアーカイブを置いた別の場所（URL、`file://`、パス）を指定できる。
手で展開した CI の artifact を試すときなど。

以下の節は、同じことを手で行う手順。

## ビルド

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss

# このフォルダ専用の Python 環境
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil

# エンジンが固定している llama.cpp のソース（setup.py の LLAMA_CPP_COMMIT）: MMQ カーネル（ggml-cuda）。ツールはもう gguf-py を使わない
C=3cf03257f219afbe7334045ff7c6a06ac68c627d
curl -L -o /tmp/llama.zip https://github.com/ggml-org/llama.cpp/archive/$C.zip
unzip -q /tmp/llama.zip -d third_party && mv third_party/llama.cpp-$C third_party/llama.cpp

# エンジン: build/abyss（v1.3.164 から。横の build/strata は旧名で、当面は別名として残す）
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
```

ROCm の `hipcc` が入っていれば、`build/libabyss_igpu.so`（下の「iGPU レーン」）も一緒にビルドされる
（対象は既定で `-DSTRATA_IGPU_ARCH=gfx1151`。`-DSTRATA_IGPU_LANE=OFF` で作らない）。

初回のビルドは数分。変更後は変わった部分だけを再コンパイルする。

> [!IMPORTANT]
> RTX 50（と RTX PRO の Blackwell）では `120` ではなく `120a` を使う。MXFP4 のカーネルが Blackwell 専用の FP4 MMA
> 命令を使い、`120` ではコンパイルできない（`ptxas error : Instruction 'mma with block scale' not supported on .target
> 'sm_120'`、issue #40）。DGX Spark は `121a`。

## モデルを取る

unsloth の [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) の `UD-Q2_K_XL`
（3 ファイル、96.8 GB）。Hugging Face の CLI（`pip install huggingface_hub` で `hf` が入る）で取る:

```sh
M=~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF   # Abyss の隣の Abyss-data（NVMe ならどこでもよい）
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf --local-dir $M
sha256sum $M/UD-Q2_K_XL/*.gguf
```

| ファイル | SHA-256 |
| --- | --- |
| `-00001-of-00003.gguf` (5 MB) | `de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b` |
| `-00002-of-00003.gguf` (49.4 GB) | `c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327` |
| `-00003-of-00003.gguf` (47.4 GB) | `a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4` |

ダウンロードが途中で止まって再開しないときは、
`curl -L -C - -o <ファイル> https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-Q2_K_XL/<ファイル>`
で続きから取れる。

同じリポジトリのほかの量子化も、エキスパートの型が decode 用カーネルの対応範囲（IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K〜Q6_K, MXFP4。IQ1_M はまとめ処理のカーネルがないので prompt は 1 トークンずつ）
なら動くはず。確認したのは UD-Q2_K_XL だけ。

ルーターや小さい射影が量子化されずに F16 で入っている DeepSeek V4 の GGUF（antirez の deepseek-v4-gguf、huihui-ai の
abliterated 版）も pack して動く。`--compat-bf16` がそれらをエンジンの GEMV が読む BF16 に丸める。huihui-ai の Q2 は
同じファイルの llama.cpp と照合した（3,000 トークンで KL 0.0075）。

## pack を作る

pack はエンジンが使う GGUF の索引（小さいテンソルの dense.bin、エキスパートの表、チャットテンプレート付きの
トークナイザー）。シャードと同じフォルダに作る:

```sh
cd ~/llm/apps/Abyss
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL/pack-dsv4 --compat-bf16
```

`--compat-bf16` は、GGUF に F32 で入っているハイパーコネクションの行列を、エンジンが読む BF16 に丸める。数秒で終わる。
エキスパートは GGUF のまま使う。

### 軽い dense pack（attention の射影を Q6_K か Q5_K に）

decode は 1 トークンごとに dense の重み全部（UD-Q2_K_XL で 6.6 GB）を読み、そのうち 4.6 GB が各層の Q8_0 の 3 つ
`attn_q_b`、`attn_output_a`、`attn_output_b`。`tools/requant_dense.py` はこの 3 つだけを再量子化し（llama.cpp のビルドの
`ggml_quantize_chunk` を使う。llama-quantize は i-quant のモデルを imatrix なしでは受け付けない）、シャードの他のバイトは
そのまま写す。できた GGUF を `iq_pack.py` で元と同じように pack にする:

```sh
cd ~/llm/apps/Abyss
LIBGGML=~/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so \
  .venv/bin/python -I tools/requant_dense.py $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  $M/UD-Q2_K_XL-attn-q6k q6_k 'blk\.\d+\.attn_q_b\.weight' 'blk\.\d+\.attn_output_[ab]\.weight'
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL-attn-q6k/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL-attn-q6k/pack-dsv4 --compat-bf16
```

90 GB のシャードを書き直すので（NVMe で 10〜25 分）、サーバーを止めてから実行する。できたら設定の pack と tokenizer の
パスを新しいフォルダに向ける。3 つの型の品質と得るもの（2026-10-15、v1.3.18。KL はエンジン自身の Q8_0 の基準 logits
との比較、4,000 トークンの最後の 1,000 トークン、CPU レーン、3 つとも同じ条件）:

| attention の射影 | 3 テンソルの大きさ | KL（基準比） | max KL | top token 一致 | nll | VRAM の expert 枠（RTX 5090） |
|---|---|---|---|---|---|---|
| Q8_0（配布された GGUF のまま） | 4.28 GiB | 0.0060 | 0.11 | 97.0% | 1.3071 | 3,010 |
| Q6_K | 3.31 GiB | 0.0082 | 0.31 | 95.8% | 1.3116 | 3,139 |
| Q5_K | 2.77 GiB | 0.0119 | 0.57 | 95.9% | 1.3135 | 3,225 |
| Q4_K | 2.27 GiB | 0.0246 | 0.84 | 92.9% | 1.3224 | 3,268 |

- Q6_K: GEMV が小さくなる分（v1.3.18 の分割カーネル）で decode +3.5%、VRAM の枠が 129 増える。swe-bench の L7（16 問 × 3、
  RTX 5090 + Radeon 8060S）では 91.3 → 93.9 tok/s で、resolved 率（81.2%）と品質スコア（81.0 → 81.2）は同じ。KL の増分は
  実行ごとのばらつきと同じ大きさ
- Q5_K: KL は Q8_0 の 2 倍（1 トークンの最大のずれ 0.57）。RTX 5090 では Q6_K より decode が 4% 速い（1,024 トークンの
  答え: 単発 12.13 → 11.65 ms、検証窓 15.52 → 14.86、86〜89 → 93 tok/s。窓も単発も同じだけ縮む）。帯域の低いカードほど
  効く: Radeon AI PRO R9700（640 GB/s）では 1 トークンの読みが 1.5 GB 減る。同じ swe-bench で 99.0 tok/s、resolved 87.5%
  （14/16）、成功率 83.3%、品質 82.0 で、KL が Q6_K の 2 倍でも劣化は見えない。**作者が常用しているのはこの pack で、
  「メモリと速度」の生成の数値もこれで測っている**（v1.3.20）
- Q4_K: 割に合わなくなる所。KL がさらに 2 倍、最頻トークンの一致が 3 ポイント落ちる（Q8_0 → Q5_K は 1.1）のに、
  単発は Q5_K より 2% しか速くならない（11.67 → 11.40 ms、枠 +43）。swe-bench は回していない
- 触る価値がないもの: `attn_q_a`、kv / compressor / indexer の射影、norm は小さく、共有エキスパート・埋め込み・出力
  ヘッドはもう Q5_K / Q6_K / Q4_K

## 設定と起動

設定ファイル（名前は自由。ここでは `strata-dsv4.json`）を Abyss のフォルダに置く:

```json
{
 "exe": "/home/you/llm/apps/Abyss/build/abyss",
 "args": ["--glm-pack", "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4",
          "--max-context", "65536", "--prefill", "auto"],
 "cwd": "/home/you/llm/apps/Abyss",
 "tokenizer": "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4/tokenizer",
 "model_name": "deepseek-v4-flash",
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082,
 "log": "/home/you/llm/apps/Abyss/strata-dsv4.log"
}
```

すべての項目、セットアップがこのファイル（`abyss-<名前>.json`）を置く場所、書き換え方は [SETTINGS.ja.md](SETTINGS.ja.md) にあります。

サーバーを起動する:

```sh
.venv/bin/python serve/server.py --engine strata --config strata-dsv4.json --port 8082
```

読み込みが終わると `ready: http://127.0.0.1:8082/v1` と表示される。起動には 1 分ほどかかる（密な重みを GPU に載せ、
全エキスパートを VRAM と RAM に読み込む。NVMe から 4 GB/s で約 20 秒）。Ctrl+C で止まる。

サーバーのオプション: `--host 0.0.0.0` でほかのマシンからの接続を受け付ける（必ず `--api-key <secret>` も付ける）、
`--open` でブラウザを開く、`--gpu 0` で GPU を選ぶ。

**どのカードで動くか。** GPU の番号は PCI バスの順で、カードを付けたり外したりすると変わる（バス番号の若い USB4 / OCuLink の
カードが先になる）。設定の `"gpu"`（と `--gpu`）には、カードの UUID（1 枚に決まれば先頭だけでよい）も書ける:
`"gpu": ["GPU-88044da7"]`（UUID は `nvidia-smi -L` で表示される。AMD は `rocm-smi --showuniqueid`）。
`build/abyss --list-gpus` は、エンジンから見える全カードの番号・UUID・PCI アドレス・名前を表示する。サーバーは起動のたびに
これで UUID を今の番号に直す（どのカードにもない UUID なら、見つかったカードを示して起動を止める）。エンジンのログには、
使うカードごとに `CUDA0 = NVIDIA GeForce RTX 5090 (UUID GPU-88044da7, PCI 0000:c5:00.0)` の行が出る。セットアップは今のところ番号を書く。

## 使い方

- **ブラウザ:** `http://127.0.0.1:8082/` でチャット、エキスパートの配置と GPU のモニター、設定。
- **OpenAI 互換 API:** ベース URL `http://127.0.0.1:8082/v1`、モデル名は何でもよい。

  ```sh
  curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
  ```

  回答は `content`、思考は `reasoning_content` に入る。
- **Anthropic 互換 API:** `http://127.0.0.1:8082/v1/messages`（Claude Code なら `ANTHROPIC_BASE_URL=http://127.0.0.1:8082`。全部の設定は `./abyss.sh --connect claude` で表示）。
- **思考:** DeepSeek V4 は指定しなければ考えてから答える。`reasoning_effort`（またはチャットの設定）で変えられる:

  | レベル | DeepSeek に渡るもの |
  | --- | --- |
  | 指定なし | 思考あり |
  | `none` | 思考なし（prompt の末尾が `</think>`） |
  | `low` | 思考あり |
  | `high` | 思考あり + DeepSeek の「Reasoning Effort: Absolute maximum」の指示 |
  | `max` | 思考あり + 「Beyond maximum」の指示 |

  DeepSeek V4.1 1017 は、自分のテンプレートがレベルをそのまま受け取る（v1.3.117）: `low` / `medium` / `high` / `max`
  （テンプレートの effort 50 / 62 / 75 / 100）か、1〜100 の整数。`none` は上と同じく思考なし。ほかのモデルには近いレベルが渡る。

  サーバーは回答に最低 1024 トークン（または `max_tokens` の 4 分の 1）を必ず残す。`max_tokens` が小さい（数百）と、
  思考はほぼすぐに打ち切られる。思考させるときは `max_tokens` を数千にする。
- **ツール呼び出し:** OpenAI の `tools`、Anthropic の `tools` が使える。1 回の応答で複数の呼び出しも可。モデルは
  DeepSeek の DSML 形式で書き、サーバーがそれを `tool_calls` に変換する。過去の呼び出しと `tool` の結果は
  prompt に描画し直す。ツールのスキーマで `boolean`（`string` を含まない）とされた引数は、thinking off のモデルが
  よく書く `True` / `False`（大文字小文字は問わない）も受け付け、JSON の `true` / `false` で返す（どのモデルの形式でも）。
- **メッセージ中の特殊トークンの文字列は文字のまま:** メッセージ、ツールの結果、ツール呼び出しの引数の中にある
  `<|im_end|>`、`</think>`、`<tool_call>`、`<｜deepseek_image｜>`（貼り付けたソースコードやログ）は、制御トークンではなく
  その文字列としてモデルに渡る。トークンになるのはチャットテンプレート自身が書く記号だけ。ただし過去の回答に含まれる
  モデル自身のタグ（`<think>...</think>`、`<tool_call>` の文字で書かれた呼び出し）は、モデルが書いたトークンのまま。
  `/api/tokens` も同じ数え方。
  チャットテンプレートが拒否したリクエストはテンプレートのメッセージ付きの `400`、それ以外のサーバー側の失敗は理由付きの
  `500` を返す（以前はどちらも応答なしで接続が切れていた）。`STRATA_RAW_SPECIALS=1` で以前の encode に戻る。
- **会話の再利用:** 前のリクエストの続き（同じメッセージ + 新しいメッセージ）は、新しいトークンだけを読む。同じプロンプトをもう一度送ると、最後のトークンだけを読む（V4.1：5K と 23K のトークンで、最初のトークンまで 0.04 秒）。
  残すのは直前の会話 1 つだけで、間に別のリクエストが入るか、再起動すると置き換わる。トークンは先頭から一致している必要があり、システムプロンプト・推論レベル・ツールの一覧が違う場合や、リクエストごとに変わるもの（タイムスタンプなど）が入る場合は、プロンプトを読み直す。再利用されたかは `usage.prompt_tokens_details.cached_tokens`（OpenAI）、`cache_read_input_tokens`（Anthropic）、ストリームの `timings.cache_n` で分かる。Ornith は v1.3.65 から再利用する。
- **リクエストは 1 つずつ**処理する。ほかは順番待ち。
- **別のプログラムからサーバーを止める**（Abyss Terminal、スクリプト、エンジンが暴走したときに Tailscale 越しのスマホから）：
  JSON の本文を付けて `POST /api/shutdown`。

  ```sh
  curl -X POST http://127.0.0.1:8082/api/shutdown -H 'Content-Type: application/json' -d '{}'
  ```

  処理中のリクエストは次のステップで打ち切る（Ctrl+C のように回答をまとめさせない）。`STRATA_SHUTDOWN_WAIT_S` 秒（15）以内に
  止まらないエンジン（GPU カーネルの中で固まっている等）は強制終了し、そのあとエンジンを終えてサーバーも終わる。
  `{"force": true}` はエンジンをすぐ強制終了する（保持中の会話は保存しない）。止まる前に `202` を返す。
  使える人: 設定に `api_key` があれば、そのキーを送った人。無ければこの PC 上のプログラムだけ（ループバック）で、
  他の PC からは `403`。ブラウザのページからは送れない（JSON 本文と、よその `Origin` が無いことが必要）。
  設定の `"allow_shutdown": false` で無効にできる。**Tailscale:** `"host": "0.0.0.0"`（か PC の Tailscale アドレス）
  なら他の機器はループバックではないので、スマホやノート PC から止めるには `api_key` を設定する。`tailscale serve`
  （やこの PC 上のリバースプロキシ）の後ろでは全員がループバックに見えるので、そこでも `api_key` を設定する
  （しないと tailnet の誰でも止められる）。
- **`/metrics` がモニター向けに返すもの**（Web アプリの Monitor、Abyss Terminal）: 量子化付きのモデル名
  （`engine.display_name`、`engine.quant`、`engine.config`、`engine.backend`）、`hardware.gpus` に全カードとその
  `role`（`main`、`split`、`helper`、`igpu` = iGPU レーンの Radeon、共有メモリ）と `source`（答えた読み取り方法:
  `nvml`、`nvidia-smi`、`sysfs`、`gpu_metrics`、`amd-smi`、`rocm-smi`、`pdh`）、クロック、ジャンクション/メモリ温度、
  スロットル理由。AMD の PCIe リンクはルートポートまでで一番狭いリンクで、`--bench` と同じ（OCuLink の R9700 は
  カード内の x16 ではなく Gen4 x4）。RDNA 3/4 には PCIe 転送量のカウンタが無いので `pcie_rx_mb` は空。モデルと
  `STRATA_GLM_MIRROR` のドライブ毎の MB/s（`hardware.drives`）、スワップ・ページキャッシュ・メモリ圧（Linux PSI）、
  CPU のコア毎の負荷・クロック・クラスタ、温度、電力、命令セット（`cpu_*`）、リクエスト毎と `totals` のドラフト
  受理数/提示数。読めない値は `null`（`--` と表示）で、0 にはしない。CPU の電力は RAPL から読むが、多くの Linux
  ディストリビューションでは root しか読めない: `sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj`
  （次の起動まで）で出る。AMD の APU は自身の電力テーブルで代わりに出す。

### 回答の言語をそろえる（`lang`）

日本語で聞いたのに、回答に中国語の単語が混ざることがあります（「误って」「这样的」など）。そのトークンがいちばん確率の高い候補だった、というだけのことです。リクエスト（`/v1/chat/completions` か `/v1/messages`）に `"lang": "<プロファイル>"` を付けると、そのプロファイルが許さない文字体系で書かれたトークンを、サンプラーの段階で禁止します。禁止したトークンは、通常の1トークンでも、サンプリングでも、下書き（MTP、DSpark、n-gram のウィンドウ）でも選ばれません。重みは変えません。`lang` を付けなければ、出力はこれまでとビット単位で同じです。

| プロファイル | 回答に使ってよいもの |
|---|---|
| `ja-en` | ラテン文字、かな、JIS X 0213 にある漢字（日本語の漢字）、ギリシャ文字 |
| `zh-en` | ラテン文字、すべての漢字、ギリシャ文字（かな・ハングルは不可） |
| `ko-en` | ラテン文字、ハングル、ギリシャ文字（漢字・かなは不可） |
| `en` | ラテン文字とギリシャ文字だけ |
| `zh-Hans-en`, `zh-Hant-en` | 試験的：`zh-en` から Big5 だけ / GB2312 だけにある漢字を除いたもの |

数字、句読点、記号、空白、絵文字、コードは、どのプロファイルでも使えます。特殊トークン（`<think>`、`<tool_call>`、ターンの終わり）は禁止しません。`"lang": "off"` で、サーバーの既定をそのリクエストだけ外せます。

- **既定では回答だけ。** 思考はそのままにします（別の言語で考えて、回答はこちらの言語で書くモデルもあります）。`"lang_think": true` で思考にも同じ制限をかけます。
- **サーバーの既定：** 設定ファイルの `env` かシェルで `STRATA_LANG=ja-en`（と `STRATA_LANG_THINK=1`）を指定すると、指定のないリクエストすべてにかかります。
- **`logit_bias`**（OpenAI 形式：`{"<トークンID>": -100..100}`、-100 で禁止）にも対応します。出力全体にかかります。
- **DeepSeek V4 0731 と日本語：** V4 0731 は、日本語で聞いても中国語で答えることがあります（元のモデルでも起きます）。プロファイルでは止まりません（日本語と共通の漢字だけで中国語を書きます）が、システムプロンプト 1 行で止まります。そこでサーバーは、システムメッセージがなく、最後のユーザーのメッセージに仮名があるリクエストに、システムメッセージ `日本語で答えてください。` を足します（V4 0731 だけ。V4.1 は対象外）。日本語 8 問 × seed 6 つ（48 回答、温度 0.6）で、中国語になった回数は、元のモデル（UD-Q8_K_XL）が 14 → 1、2 ビットのファイルが 16 → 1 でした。`ja-en` だけでは 12 / 48 です。`STRATA_JA_SYSPROMPT=0` で止められます。自分のシステムメッセージがあるリクエストはそのままです。
- **副作用の測定：** 手元にある正しい日本語の文章 1.3 MB（約 20 万トークン）で、`ja-en` が禁止トークンを必要としたのは 2 回だけでした（キリル文字・アラビア文字を使った顔文字 2 つ）。両性具有のような日本語の単語は影響を受けません。JIS X 0213 の漢字は 1 つも禁止しないからです。トークナイザーごとの分類表は `data/token_classes/` にあり、`tools/token_classes.py` で作ります（自分の文章で副作用を測る `side-effects` もあります）。

### Abyss Terminal（`./abyss.sh --terminal`）

サーバー（この PC のものでも別の PC のものでも）を見るターミナル UI。開いても閉じてもモデルは止まらない。

```sh
./abyss.sh --terminal                       # 動いているサーバー、無ければ最後に使った設定（Enter で起動）
./abyss.sh --terminal --config abyss-dsv41-iq2xxs-r9700l.json
ABYSS_API_KEY=... ./abyss.sh --terminal --url http://192.168.1.20:8082 --key-env ABYSS_API_KEY   # 別の PC
./abyss.sh --terminal --plain               # Textual 無し: 1 秒毎に書き直すプレーン表示（古い Windows のコンソール）
./abyss.sh --terminal --once                # 1 回読んで表示、--json は JSON で（issue 報告用）
./abyss.sh --terminal --demo r9700          # 記録した値でサーバー無しに表示（r9700, lowspec, split, multi4, multi8,
                                            #   multi10, loading, starting, down）
./abyss.sh --terminal --layout devices      # Overview をデバイス毎のパネルで（3 枚以上のカードでは既定）
```

Windows は `START-ABYSS.bat --terminal`（オプションは同じ）。abyss.sh を使わない場合は `python tools/abyss_terminal.py ...`。
画面は Textual で、初回に `third_party/wheels` から `.venv` へ入る（ダウンロードはしない）。

- **ビュー**（`1`-`8`、`←` `→`）: **Overview** - decode と prompt の tok/s を大きく（動作中はライブ、待機中は前の
  リクエストの値とラベルで区別）、1 トークンの時間の行き先を 1 本のバーで（SSD 待ち / GPU、CPU・iGPU レーンは並行
  なので別線で示し足さない）、`--bench` と同じ判定と助言（同じコード `tools/bench_rules.py`）、一番忙しいカード、
  VRAM、一番熱いカード（ジャンクション温度も）、電力、作業中の PCIe リンク、CPU、ディスク（`STRATA_GLM_MIRROR` の
  SSD 毎）、コンテキスト、VRAM のエキスパート、RAM tier、スワップとメモリ圧込みのシステム RAM、ドラフタ受理率、
  全カード（main / split / helper / iGPU レーン）、最近のリクエスト。**Speed** - 内訳と判定の根拠、ドライブ毎の SSD
  読み出しと待ち、ミラーの分担、レーン、ドラフタ、最後の `--bench` と今の比較。**Memory** - エキスパートの居場所、
  KV、RAM、「何が遅くしているか」。**Requests** - 保持中の全リクエスト（`↑` `↓`: Terminal を開いてから答えたものは
  エンジンの 1 トークン内訳も）。**Hardware** - カード毎のクロック、edge / junction / メモリ温度、スロットル理由、
  PCIe、CPU のコア・クラスタ毎、ドライブ。**Logs** - 色分けしたエンジンログ（`/` で絞り込み）。**Connect** - API の
  アドレスと疎通確認（`h`）。**Models** - インストール済みの設定: `Enter` で起動（動いているサーバーは先に止める）、
  `s` で停止、`B` で `--bench`（出力は Logs）、`y` でベンチ要約行をコピー。
- **Overview の 2 つのレイアウト**（`l` で切替、`--layout speed|devices`。選んだものは `~/.config/abyss/terminal.json`、
  Windows は `%APPDATA%\abyss\terminal.json` に覚える）: **Speed** - 上の大きな数字、1 トークンの時間、タイル。
  **Devices** - 小さな速度の帯の下に、GPU（main / split / helper / iGPU レーン）、CPU、ドライブを 1 台ずつのパネルで
  折り返して並べる（10 枚まで収まる）。まだ選んでいなければ、カード 3 枚以上の設定は Devices で開く。フッタに今の
  レイアウト名が出る。
- **複数の GPU**（2-10 枚、世代の混在も）: カード毎に役割、負荷、VRAM、レイヤ分割で受け持つレイヤとエキスパート
  プール（`L0-13`、この PC のエンジンログから）、温度、そのカード自身の上限に対する電力、そのカード自身の最大に対する
  PCIe リンク、1 トークンの時間の分担（エンジンの STAT 行から。v1.3.80）。一番忙しいカードに ★、作業中にリンクが落ちたカードや
  他の半分未満のリンク（x1 ライザー、Gen2/Gen3 x4）のカードはピンクでバナーにも出る。ドライブは何台でも MB/s と
  エキスパート読み出しの分担を出す。80x24 ではカードを 1 行に要約（一番忙しい / VRAM / 一番熱い / 一番遅いリンク）し、
  要るカードだけ並べる。
- **デタッチ:** Terminal が起動したサーバーは別セッションで動く（`abyss.py --config <設定> --plain` を
  `ABYSS_NO_BROWSER=1` で、ブラウザは開かない。出力は `<設定>.server.log`、PID は `<設定>.pid`）。終了時に「提供を続ける / 止める」を聞く。既に動いていたサーバーには
  触らない。停止は `POST /api/shutdown`、使えなければサーバーのプロセスグループへ Ctrl+C、60 秒で強制終了。次の
  `--terminal` でまた繋がる。
- **ヘッダ**は必ず量子化付きでモデルを示し（例 *DeepSeek V4.1 Flash 1017 UD-IQ2_XXS*）、答えが変わる設定をバッジで
  出す（`UNCENSOR`、`TAIL SKIP n`、`DEFER`、`ABL`。`MTP`、`MIRROR`、`HELPER`）。状態: 接続中、ロード中（エンジンログ
  からの warm-up 進捗）、待機、プロンプト読込、生成中、待ち行列、リロード、エラー、STALE（最後の値を保持）、切断
  （間隔を空けて再試行）、認証が必要、Abyss ではない、停止。
- **1 トークンの時間をライブで（1.3.80 以降のエンジン）:** エンジンの STAT 行（8 トークン毎）に SSD 待ち、レーンの
  エキスパート数と busy ms、ミス、tail skip、ミラーのドライブ毎 MB、ドラフタの受理 / 提示 / 検証ステップ数、直近
  約 64 トークンの窓が乗る。回答を書いている間は Token time タイルと Speed ビューがその窓を出し（`live · last ~64
  tokens`）、`--bench` と同じ判定がそれに追従する。待機中は最後の回答の最終値（`last answer · engine STAT`）、古い
  エンジンでは回答毎のエンジンログ（`engine log`）。Speed ビューにはレーンの推定帯域（1 トークンのエキスパート数 ×
  1 エキスパートの MiB ÷ busy ms）、エンジンが数えたミラーの分担、ドラフタのライブ受理率と 1 検証ステップ当たりの
  トークン数が加わり、プロンプト読込中は prompt タイルに再利用トークン数とここまでの SSD 待ちが出る。ヘッダのバッジ
  （`TAIL SKIP`、`DEFER`、`MIRROR`、`HELPER`、ドラフタ）はエンジンの INFO 行から取るので、リモートでも出る。
- **起動中のサーバー（1.3.80 以降）:** エンジンを起動する前にポートを開くので、`/health` が `status: "loading"` と
  段階・warm-up の % と GB・経過秒を返す。Terminal は別の PC からでもこの PC と同じように進捗を出す。v1.3.81 から、
  この応答は 503 と `Retry-After: 5` になる（200 を「準備できた」と読むクライアント、たとえば llama-swap の health check も
  エンジンを待つ）。準備ができたら 200。他のリクエストも準備できるまで 503 + Retry-After。コンテキスト変更のリロード中も同じ進捗が `reloading` の下に出る。
- **`--` は「報告されていない」**で、0 ではない。まだエンジンのカウンタ待ちなのは、ヘルパーカードのトークン時間の
  分担（E6、CUDA 2 枚が要る）。
- **リモート:** 読み取りのみ（この PC の設定ファイルを読む Models は出ない）。キーは `--key-env` で指定した環境変数
  から読み、保存しない。別の PC へ平文の `http://` で送ると読まれうる - Tailscale か SSH トンネル
  （`ssh -L 8082:127.0.0.1:8082 host` の後 `--url http://127.0.0.1:8082`）を使う。1.3.80 以降のサーバーはエンジン
  ログの末尾を `GET /api/log?since=<offset>&max=<bytes>` で送る（キーが要る。API キーの値は伏せる）ので、Logs と
  リクエスト毎の内訳もリモートで出る。古いサーバーでは Logs にその旨のヒントが出る。
- **キー:** `l` Overview のレイアウト、`r` すぐ再取得、`space` 表示を止める、`t` テーマ（dark / light / mono。`NO_COLOR` なら mono で始まる）、
  `L` 温度の CSV を 2 秒毎に `bench/thermal/` へ記録、`b` web アプリを開く、`c` API アドレスをコピー（OSC 52）、
  `?` ヘルプ、`q` / Ctrl+C で終了。80x24 で崩れず（箱を詰め、ビューはスクロール）、110 桁以上で表示が増える。

### Strata-tools の strata.sh を使う場合

`strata.sh`（start / stop / log）を使うなら、Abyss のフォルダに起動スクリプトと設定を `run-<名前>.sh`,
`strata-<名前>.json` の名前で置いて、`STRATA_DIR=~/llm/apps/Abyss ./strata.sh start <名前>` で起動する。
起動スクリプトの中身は上の起動コマンド（`exec .venv/bin/python serve/server.py --engine strata "--config" "<設定>" "--port" "8082" "$@"`）。

### コーディングツール: Claude Code と OpenCode（`--connect`）

`./abyss.sh --connect <ツール>` は、いま動いているサーバー（`GET /v1/models` のモデル名とコンテキスト）と、その起動設定
（アドレス、API キー）を読んで、そのツール用の設定を表示する。`--write` を付けない限り何も書き込まない。

```sh
./abyss.sh --connect claude              # Claude Code を起動するときの環境変数
./abyss.sh --connect claude --write      # 設定ファイルの横に run-claude-<設定名>.sh を書く（書く前に確認）
./abyss.sh --connect claude --check      # Claude Code と同じ形でプロンプト 1 回 + ツール呼び出し 1 回を試す
./abyss.sh --connect opencode            # opencode.json に入れる provider のブロック
./abyss.sh --connect opencode --write    # ~/.config/opencode/opencode.json に provider "abyss" を追加（バックアップを残す）
./abyss.sh --connect opencode --check    # 同じ確認を OpenAI 形式で
ABYSS_KEY=... ./abyss.sh --connect claude --url http://192.168.1.20:8082 --key-env ABYSS_KEY   # 別の PC のサーバー
```

- **Claude Code:** ランチャーは `ANTHROPIC_BASE_URL`、全モデル枠（`ANTHROPIC_MODEL`、Opus / Sonnet / Haiku の既定、
  `CLAUDE_CODE_SUBAGENT_MODEL`）をサーバーのモデルに、`CLAUDE_CODE_MAX_CONTEXT_TOKENS` をサーバーのコンテキストに
  （知らないモデルだと Claude Code は 200K とみなし、要約が遅れる）、`CLAUDE_CODE_MAX_OUTPUT_TOKENS`（コンテキストの 1/4、
  最大 32K）、`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`、`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` を、その起動の間だけ
  設定する。`~/.claude/settings.json` は変えない。キーは `$ABYSS_API_KEY` から `ANTHROPIC_AUTH_TOKEN`（Bearer）で送る。
  `ANTHROPIC_API_KEY` は Claude Code が使うかどうか聞いてくるので外す。`~/.claude/settings.json` の `env` に
  `ANTHROPIC_*` があるとそちらが優先されるので、消しておく。Claude Code 自身のプロンプトが約 20K トークンあるため、
  サーバーのコンテキストは 32K 以上（64K 以上なら余裕がある）で設定する。モデル名が `claude-` で始まると Claude の
  モデルと見なされるので、設定の `model_name` を別の名前にする。
  Claude Code のサブエージェントはメインスレッドとは別の会話として動く。エンジンは片方が動く間もう片方を退避して、
  戻ってきたら復元する（`STRATA_GLM_SLOTS`）ので、戻るときは新しい部分だけを読む（Qwen3.8、3 万トークンで 11 秒 → 0.2 秒）。
- **OpenCode:** provider `abyss`（`@ai-sdk/openai-compatible`、ベース URL `.../v1`、`limit.context` = サーバーの
  コンテキスト、キーは `{env:ABYSS_API_KEY}`）。既定のモデルがあればそのまま残すので、`/models` で `abyss/<モデル>` を
  選ぶ。コメント入りの設定（JSONC）は書き換えないので、表示されたブロックを手で貼る。
- **コーディング・エージェントには温度 0 も選べる（任意）。** 設定の既定はモデルカードのサンプリングのまま（例: Qwen3.8 は
  temperature 1.0、top_p 0.95、top_k 20）。コーディングやエージェントの作業では、`abyss-<名前>.json` に `"sampling":
  {"temperature": 0.0, "top_p": 1.0, "top_k": 1}` と書けます。いつも一番確率の高いトークンを選ぶので、ドラフターの下書きが
  当たりやすくなります。RTX 4090 の Qwen3.8-Flash-Next で、下書きの採用率が 76 → 86%、長いコーディングのプロンプトが約 20% 速く
  なりました（[#39](https://github.com/zephel01/Abyss/issues/39)）。自由な文章では同じ言い回しを繰り返すことがあるので、会話や
  文章書きには既定のままがおすすめです。リクエストの `temperature` は設定より優先されます。
- キーはファイルに書かない。Claude Code 2.1.316 と RTX 5090 1 枚の Qwen3.8-35B-A3B-Distill で、ファイルを書くツール呼び出しまで確認した。

## 複数の GPU

NVIDIA の GPU が 2 枚以上あれば、層を分けて載せられる（layer split）。各 GPU は自分の層のエキスパートだけを VRAM に置く。
2 枚目の VRAM の分だけ、VRAM に載るエキスパートが増える。

- **セットアップ:** `./setup-deepseek.sh --gpus 0,1` または `--gpus all`。設定に `"gpu": [0, 1]` が入り、サーバーが
  `--layer-split auto` を付けて起動する。世代の違う GPU（例: RTX 5090 + RTX 3090）は、両方のアーキテクチャ
  （`86;120a`）でビルドする。
- **`auto` の選び方（v1.3.25）:** すべての境目と「各カード 1 枚だけ」を、予測した decode 時間で比べる（各カードの読む
  バイト ÷ メモリ帯域、CPU / iGPU レーンが隠しきれないミスの分、境目 1 つごとの固定コスト。RTX 5090 + RTX 3090 で
  合わせた係数で、実測 10.9〜18.1 ms との差は 0.35 ms 以内）。1 枚のほうが速ければ 1 枚で動かす。decode の 1 トークンは
  part を順番に通るので、遅い 2 枚目の層は、増えた VRAM で減るミスより高くつくことがある。RTX 5090 + RTX 3090（どちらも
  PCIe Gen4 x4）では 5090 だけで 94〜95 tok/s（以前選んでいた 26 / 17 の分割は 84 tok/s、最良の分割 33 / 10 でも
  88 tok/s）。`STRATA_GLM_SPLIT_KEEP=1` で分割を続ける（長い文脈や prompt 用の VRAM が欲しいとき）。
  `STRATA_GLM_SPLIT_LOG=1` で予測をすべて出す。
- **境目を手で決める:** 設定の `"layer_split": "22"`（2 枚目が担当する最初の層）。
- **iGPU レーン:** 分けたときも使える。各 GPU の RAM 層を 1 つの iGPU が順番に計算する。
- **確認済みの範囲:** 1 枚の GPU の上で 2 つに分けた形（`STRATA_GLM_SPLIT_SAME=1 STRATA_GLM_SPLIT=22
  STRATA_GLM_DEVS=0,0`、試験用）で、prompt と decode の結果が分けない場合と一致すること（KL がぶれの範囲内）。
  RTX 5090（PCIe Gen4 x4）+ RTX 3090（USB4 ケース、約 3.8 GB/s）を 27 / 16 層に分けた実測は、iGPU レーンありで
  prompt 8K / 32K が 1,506 / 1,985 tok/s、decode が 54.9 / 56.7 tok/s（RTX 5090 1 枚 + iGPU: 1,683 / 2,876 と
  58.5 / 65.9）。2 枚目で VRAM の当たり率は 82 → 91% に上がるが、カード間の受け渡しと遅いリンクのコストのほうが
  大きかった。PCIe に直接挿した速い 2 枚目ではまだ測っていない。
- **投機デコード**（v1.3.26）: 検証窓を part ごとに順に通し、各 part の列を 1 トークンのときと同じように次の part に
  渡す。RTX 5090 + RTX 3090 を 33 / 10 に分けて 87〜90 tok/s（なしで約 80〜88）、候補の採用率は 1 枚のときと同じ
  （コードで 57%）。
- **prestage**（v1.3.27）: 各 part が、次の層で使いそうな RAM 層のエキスパートを mixer の間に VRAM へ送る（1 枚の
  ときと同じ）。分割では chunk の大きさを先に決め（最大 8192）、残った分を prestage に使う。RTX 5090 + RTX 3090 の
  33 / 10 分割で、32K の prompt が 2,177〜2,186 → 2,274〜2,350 tok/s、8K はほぼ同じ。`STRATA_GLM_PRESTAGE=0` でオフ。
- **Qwen3.8-Flash-Next（qwen4exp、v1.3.109 からベータ）:** 分割したまま、トークン、prompt、検証窓が動く。MTP ドラフター（`STRATA_Q4X_MTP`）は
  最後の part に置く（最後の残差と head を使うため。その part も token_embd の pinned コピーを持ち、RAM を 497 MB 多く
  使う）。最初の境目は層 2 以降（層ごとの埋め込みは最初の part が足す）。`auto` は、文脈ぶんの delta-net の状態、K / V
  キャッシュ、QSA の鍵と、最後の part のドラフターを数える。RTX 5090 1 枚を 2 つに見立てて確認した（M16b 版、層 24 で
  分割）: KL は分けない場合と同じ、実用テストも同じ、候補の採用率 84.7%（分けない場合 83.2%）。本物の 16 GB + 16 GB の
  GeForce 2 枚（#35）では、層 32 で分けると長いリクエストで 22% 速かったが、そこでも `auto` は 1 枚目だけを選ぶ（この数字で
  コストのモデルを合わせ直す予定）。それまでは `--layer-split 32` を手で指定する。v1.3.110 から、手で決めた分割はモデル
  自身の層の数で確かめる（それまでは、48 層のモデルで `--layer-split 46` が断られていた）。`auto` が 1 枚で動かすときは、
  そのカード自身の `STRATA_GLM_VRAM_GB_PARTS` の値を使う。2 枚で動かした方の報告を歓迎する（ベンチの issue フォーム）。分割中は会話の使い回し
  （スナップショット）がオフになる。
- **境目での受け渡し**（試験用）: `STRATA_GLM_HOP=host`（既定。pinned のホストバッファ）、`peer`（デバイスからデバイスへ。
  peer アクセスのないカードは host になる）、`sock`（バイト列。別のプロセスや別の PC へのつなぎと同じ形）。RTX 5090 1 枚を
  2 つに見立てると、3 つとも KL は同じで、44.6〜46.1 tok/s。
- **part ごとの VRAM の上限**（試験用）: `STRATA_GLM_VRAM_GB_PARTS=14,14` で、part の順に、それぞれの
  `STRATA_GLM_VRAM_GB` を決める。`auto` も同じ上限で見積もる。
- AMD の GPU は下の「AMD の GPU」を参照。
- **V100（sm_70）:** CUDA 13 は V100 向けにコンパイルできないので、CUDA 12（12.9）を入れて
  `./setup-deepseek.sh --nvcc /usr/local/cuda-12.9/bin/nvcc` でビルドする（CUDA 12.9 は RTX 50 の `120a` も作れる）。
  sm_70 でのビルドは通ることを確かめた。V100 の実機ではまだ動かしていない。tensor core を使う新しいカーネル
  （indexer、prompt の attention）は sm_80 以上が条件で、V100 では従来のカーネルを使う。

## AMD の GPU（HIP、試験的）

AMD の Radeon でも動く。対象は RX 7900 XT / XTX（gfx1100）、RX 9070 / AI PRO R9700（gfx1201）、Ryzen AI Max の
Radeon 8060S（gfx1151）。

- **セットアップ:** `./setup-deepseek.sh --backend hip`（NVIDIA の GPU がなければ自動で HIP を選ぶ）。Linux では
  `/opt/rocm`（または `/opt/rocm/core-10.0`）の ROCm 10 を使う。HIP 7.14 以降が要り、ROCm 7.0〜7.2 では止まる。この
  2 つはいまのドライバと組み合わせると失敗するため（issue #1、#8。`ABYSS_OLD_ROCM=1` でそれでもビルドできる）。
  エンジンは `build-hip/abyss` にビルドされ、設定に `"backend": "hip"` が入る。別の ROCm に切り替えたら、
  `ROCM_PATH` を付けてもう一度実行する。その ROCm でエンジンを最初からビルドし直す（v1.3.44）。Windows で
  Visual Studio の MSVC が 14.50 以降なら、セットアップが ROCm SDK の `__clang_hip_runtime_wrapper.h` の include を
  1 行移す（「'isgreater' cannot overload」への LLVM の修正。issue #43。元は `.abyss-orig` として残る）。
  `ABYSS_NO_ROCM_HEADER_FIX=1` で SDK に手を付けない。
- **ドライバと ROCm:** 入れ方は [docs/amd/README.ja.md](amd/README.ja.md) にまとめた（amdgpu、ROCm 10 の
  アーキテクチャ別のパッケージ、`HSA_OVERRIDE_GFX_VERSION` を設定してはいけない理由）。動作確認とトラブル対応も
  そこにある。R9700 では `amdrocm10.0-gfx1201` が要る。
- **確認済みの範囲:** Radeon 8060S（gfx1151）単体と、AI PRO R9700（gfx1201）で 8060S を iGPU レーンにした構成。
  R9700 では DeepSeek V4 が swe-bench の問題で 36〜42 tok/s（v1.3.39）、51.7 tok/s（v1.3.41。レーンなしで 49.6）、
  1 トークンずつで 52 tok/s（v1.3.40。レーンなしは v1.3.41 で 46）で、CUDA 版との KL はぶれの範囲内（0.005〜0.006）
  だった。DSpark のドラフトモデル（v1.3.43）を使うと、同じ swe-bench の問題がレーンなしで 61.9 tok/s、レーンありで
  67.3 tok/s（[DSpark のドラフトモデル](#dspark-のドラフトモデルdeepseek-v4-flash)）。
- **ROCm の HW キュー:** `GPU_MAX_HW_QUEUES` は 2 のままにする（セットアップが書く）。R9700 向けの手順にある 1 では、
  prompt が 3 分の 1 ほど遅くなる（[docs/amd](amd/README.ja.md#3-環境変数)）。
- **Ryzen AI Max に単体の Radeon をつないだ場合:** NVIDIA のカードのときと同じく、iGPU が RAM 層のエキスパートを
  計算する（R9700 で生成 +12〜46%）。設定の `"gpu_pci"` があれば、HIP の番号が変わっても同じカードを使う。
- **速くなる部分:** indexer のスコア計算と prompt の attention は RDNA の WMMA を使う（gfx11 / gfx12）。
  `STRATA_DS4_IDX=1` で indexer を、`STRATA_DS4_ATTN=1` で attention を従来のカーネルに戻せる。
- **制限:** NVIDIA との組み合わせ（1 つのプロセスで CUDA と HIP の両方）には対応していない。

### この PC を測る（`./abyss.sh --bench`）

`./abyss.sh --bench`（Abyss を止めて実行、数分）は、入っているモデルをどの PC でも同じ方法で測ります。3 つの質問での生成速度、2k / 8k トークンのプロンプトを読む速度に加えて、1 トークンの時間がどこで使われているか（SSD 待ち、CPU / iGPU レーン、PCIe での転送、GPU）、一番ありそうなボトルネックと速くする方法、同じモデルの正常な実行と比べた品質チェック（`bench/ref`）を出します。`--deep` は GPU の稼働率と、PCIe / レーンの分担率の振り分けを足します。`--full` は回答を繰り返し、32k トークンのプロンプトも測ります。結果は `abyss-bench.txt` に書き出すので、issue に貼ってください。各行の意味は [BENCH.ja.md](BENCH.ja.md) にあります。

### この PC に合わせてレーンを調整する（`./abyss.sh --calibrate`）

セットアップのあとに一度実行してください（Abyss を止めて、約 10〜15 分。最初にモデルを読み込みます）。RAM の枠にある
エキスパートを CPU のレーンと PCIe にどう分けるか、CPU のスレッド数をいくつか試して生成の速さを測り、エンジンの既定より
3% 以上速い設定を残します（設定の `env` の `STRATA_GLM_PCIE_SHARE`、`STRATA_GLM_CPU_LANE` と、
`~/.config/project-maya/calibration.json`。もう一度セットアップしても残ります）。一番よい分け方は PC によって違います:
RTX 4090 + RAM 96 GB の Qwen3.8-Flash-Next UD-Q4_K_XL では、割合が 0.08 → 0.10 になり、生成が 50.2 → 56.5 tok/s（+12%、
[#39](https://github.com/zephel01/Abyss/issues/39)）。GPU を替えた、RAM を増やした、モデルを替えた、設定を書き直す
セットアップをした、というときはもう一度実行してください。

### モデルが正しく動くか確かめる（`./abyss.sh --check`）

`./abyss.sh --check`（v1.3.112。Abyss を止めて実行、15〜30 分。`--quick` は 5 分ほど、`--long` はコンテキストが許す一番大きなプロンプトまで）は、入っているモデルを実際の使い方で動かし、落ちないか、答えが正しいかを確かめます。ドラフターあり・なしの答え、日本語、テストで動かす小さなコーディング課題、ツール呼び出し、8K / 24K / 32K からコンテキストの上限近くまでの長いプロンプト（埋め込んだ答えを探す）、11 回のやりとりの会話です。結果は `abyss-check.txt` に書き出します（ホスト名・ユーザー名・フォルダのパスは消し、どこにも送りません）。この PC、モデル、「この PC で何が速さを止めているか」（GPU、CPU レーン、PCIe、RAM、SSD、VRAM のどれが 1 トークンを遅らせているかの見積もり）が入ります。[バグ報告](https://github.com/zephel01/Abyss/issues/new?template=bug.md)に添えてください（`--report` にも入ります）。終了コードが 1 になるのは、失敗したステップがあるときだけです。セットアップの前、または `--setup` / `--gpu` / `--gpus` を付けたときは、これまでどおり PC だけを調べます。Windows は `START-ABYSS.bat --check`。各ステップの中身は [BENCH.ja.md](BENCH.ja.md) にあります。


### 日本の金融の規則どおりに答えるか（`./abyss.sh --check --finance`）

`./abyss.sh --check --finance`（v1.3.123。Abyss を止めて実行、10〜30 分ほど）は、入っているモデルに、架空の口座の数字で日本語の 17 問を解かせます。FX・CFD・暗号資産の証拠金、信用取引の保証金と追証、スワップポイントや配当落調整額を価格の損益と分けること、上場株式の税、NISA の枠、外貨預金の利回りです。ほかに、必要な数字（代用有価証券の掛目、ロスカットの水準、スワップの額）が問題に無いときに「わからない」と答えられるかを見る 3 問と、売り買いをはっきり勧めるよう求められても断定しないかを見る 1 問があります。どの問題も**使う規則（率、下限、税率）を問題文に書いています**。学習のあとに変わった規則を知らないことで減点しないためです。規則は `tools/finance_check.py` の 1 つの表に、出典と日付つきでまとめています（`RULES_VERSION`）。採点するのは相場に左右されないことだけで、最後の `答え:` の行で判定します（考える過程は見ません）。全部の答えと判定は `abyss-finance.txt` に書き出します。点数は「17 問中 n 問が規則どおり（規則の版）」で、投資の助言ではなく、モデルの金融の実力を測るものでもありません。OpenAI 互換のサーバーなら `python tools/finance_check.py --url http://127.0.0.1:8080` でも使えます。
### GPU に合わせた調整（`./abyss.sh --gpu-tune`）

カーネルの選び方には GPU によって速さが変わるものがある。エンジンは手元で実測した GPU（AI PRO R9700、RTX 5090）の
結果から、GPU のアーキテクチャごとに選ぶ（ログに `glm fast: dense GEMV tune` と出る）。`./abyss.sh --gpu-tune`
（v1.3.51）は、モデルなしで、その PC の GPU で 1 台 1 分以内に測る（モデルのサーバーは止めておく）:

- GPU の種類（アーキテクチャ、CU 数、VRAM）と、メモリコピーの帯域
- Q6_K の dense GEMV を 1 行 4 warp と 1 warp で比べる（モデルの行数で。R9700 は 1 warp が速い: 560 -> 510 us）

エンジンの選択より 2% 以上速いものがあれば、インストール済みの設定の `env`（`STRATA_GLM_MV_Q6K_SPLIT`）に書き込み、
次の起動から使う。結果は `abyss-gpu-tune.txt` にも保存する。この 2 機種以外の GPU なら、
[issue](https://github.com/zephel01/Abyss/issues/new?template=gpu-tune.md) に貼ってほしい。その GPU の既定値は、
こうした報告をもとに決める。v1.3.115 から、別のフォルダーで手でビルドしたエンジン（設定ファイルの `exe`。例: `build-cuda`）
も見つけるので、「not built here yet」で止まらない（#37）。

## NVIDIA DGX Spark（GB10、ARM、ベータ）

v1.3.113 から、エンジンは aarch64 + sm_121 でビルドでき、DGX Spark の GB10 で動く。GB10 は GPU と CPU が 1 つの
メモリを分け合うので、エキスパートの置き場と RAM 層を 1 つの予算から決め（`STRATA_GLM_UMA_HEADROOM_GB` を空けておく）、
programmatic dependent launch を有効にし（`STRATA_GLM_PDL`）、RAM 層のエキスパートは置き場の枠に写してから使い
（`STRATA_GLM_UMA_DIRECT=1` でその場で読む）、CPU レーンは無効にする。借りた Spark で Qwen3.8 16G64G を測ったところ、
MTP の下書きありで 47.05 tok/s（CLI、4K）、サーバー経由で 43〜54 tok/s、8K のプロンプトは約 1,040 tok/s。v1.3.114
（2026-11-06）では、GLM-5.3-Flash が 19〜20 tok/s、DeepSeek V4 0731 が 21.0（DSpark の下書きありで 22.8）、DeepSeek V4.1
Compact が約 18。GLM の長いプロンプトは、そこではサーバー経由だとまだ遅い（約 120〜130 tok/s、CLI では 675〜796。未解決）。
v1.3.115 から、ほかのプロセスが使っているメモリの上では始めない（`STRATA_GLM_UMA_SHARE`、`STRATA_GLM_UMA_OTHERS_GB`）。
ビルドの手順、測った数字、まだ試していないことは [docs/DGX-SPARK.ja.md](DGX-SPARK.ja.md) にある。

## DeepSeek V4.1 Flash

v1.3.29 から、同じエンジンで DeepSeek-V4.1-Flash-1017 が動く。`./setup-deepseek.sh --model v41`（v1.3.100）が zephel01 の
LDLQ GGUF を入れる（[まとめてセットアップ](#まとめてセットアップ)）。ほかの GGUF は手作業で準備する。
読める GGUF、pack の作り方、設定例、測定結果は [docs/deepseek41/README.md](deepseek41/README.md)、V4 から変わった計算は
[docs/deepseek41/SPEC.md](deepseek41/SPEC.md)（どちらも英語）。要点:

- **ディスクとメモリ。** nazeshinjite の Q2 GGUF は 366 GB。エキスパートが 153 GB（層ごとに 384 個、1 個 9.95 MB）、Engram
  の表が約 200 GB、残りが密な重み。RTX 5090 + RAM 94 GB では、エキスパートの約 57% が常駐する（VRAM に約 2,300 個、60 GB
  の RAM 層に約 6,500 個）。残りと Engram の行（1 トークン 48 行、1 行 264 バイト）は GGUF から読む。いちばん速い NVMe に
  置き、RAM は約 25 GB をページキャッシュに残す（94 GB の機械なら `STRATA_GLM_RAM_GB=60`）。
  60 は 94 GB の機械向けの値。RAM がもっと多い機械では RAM 層を大きくするか、変数を設定しない（空き RAM から決まる）。
  RAM 123 GB の機械（RTX 5090、GLM-5.3-Flash Maya-S-v2 IQ2_XXS、エキスパート約 80 GB）では、60 だと長いコンテキストで
  エキスパートが SSD に残り、66 にすると SSD の待ちが 1 トークン 5〜7 ms から 0.2〜2 ms に減って、生成は 60K で
  37.8 → 50.7 tok/s、125K で 40.3 → 49.5、250K で 33.3 → 42.1 になった（空きメモリは常に 44 GB 以上）。
- **速さは SSD で決まる。** SSD からエキスパートを読まないトークンは 13〜15 ms。読むたびに 3.4 GB/s で約 3〜4 ms 増える。
  `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1`（ルートの 4〜6 位が SSD にしかないときは計算しない。量子化前の
  モデルに対する KL は 0.229 → 0.240）で生成は約 28 tok/s、なしで 17.7 tok/s。swe-bench のコーディング問題は通しで
  51.6 tok/s。prompt は V4 より遅い。prompt のエキスパートを SSD から読むため（アテンションは v1.3.60 からまとめて計算する）。
- **軽い dense pack**（v1.3.36、任意）: `attn_q_b` / `attn_output_a` / `attn_output_b` を Q8_0 から Q5_K にすると、
  dense が 8.14 → 6.50 GB、VRAM のエキスパート枠が 177 増え、生成が 10〜15% 速くなる。KL は 0.242 → 0.251。作り方は
  [docs/deepseek41/README.md](deepseek41/README.md#a-lighter-dense-pack-q5_k)。
- **メモリ上限つきのスコープで起動する**（`systemd-run --user --scope -p MemoryMax=88G ...`）。ページキャッシュと pinned の
  層が重なると、systemd-oomd がデスクトップのセッションを止めることがある。
- **投機デコード**（v1.3.53、任意）: Lucebox の V4.1 用 DSpark ドラフターを、`STRATA_DS4_DSPARK=<ファイル>` で手で指定する
  （セットアップの `--dspark` が入れるのは V4 0731 用のドラフターで、これではない。Lucebox の V4.1 ドラフターのリポジトリは
  Hugging Face から消えている。v1.3.100）。n-gram 窓も使う。モデル自身の MTP 層はどの GGUF にも入っていない。
- **画像**（v1.3.93、PR #28。ISTA-DASLab の新しい形式のテンプレートは v1.3.105 から）: ISTA-DASLab の V4.1 用 mmproj と
  画像用のルーターのバイアスを手で設定する: [docs/SETTINGS.ja.md](SETTINGS.ja.md#deepseek-v41-の画像)。
- **まだできないこと:** 層の分割、会話のディスク保存。

## Qwen3.8-Flash-Next

v1.3.74 から、エンジンは Qwen3.8-Flash-Next（llama.cpp の `qwen4exp`）を速い経路（`--glm-pack`）で動かす。prompt の経路、
検証窓、MTP ドラフターも使える。準備は手作業。読める GGUF、pack の作り方（`tools/iq_pack.py --fast`）、ドラフター、
設定例、測定結果は [docs/qwen38/README.ja.md](qwen38/README.ja.md) にある。要点:

- **自前の版:** [16G64G](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF)（KL 0.083。RTX 5090 で、32 GB なら
  MTP ドラフターありで 128.7 tok/s、16 GB なら 95.1）と、
  [12G32G](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF)（L448: 512 個のうち 448 個のエキスパート、67.5 GB。
  GPU 12 GB + RAM 32 GB 向け）。
- **ほかの GGUF**（ベータ）: NVIDIA の NVFP4 チェックポイント（v1.3.101）、unsloth の UD 3.1 のセット（v1.3.112〜v1.3.115）。
  それぞれの数字は docs/qwen38 にある。
- **GPU 2 枚**（ベータ、v1.3.109）: [複数の GPU](#複数の-gpu)。**DGX Spark**（ベータ、v1.3.113）:
  [NVIDIA DGX Spark](#nvidia-dgx-sparkgb10armベータ)。
- **DFlash 2 ドラフター**（オプトイン、v1.3.125）: `STRATA_Q4X_DFLASH=<GGUF>` で、incoai の
  [DFlash 2](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF) ブロックドラフターを nazeshinjite が
  Q8_0 GGUF にしたもの（ファイル 0.65 GB、VRAM 0.76 GB）を MTP 層の代わりに読む。小さな 5 層が、モデルの 6/16/26/36/46 層に
  入る残差を読み、1 回で最大 5 トークンを下書きする。検証窓が MTP の下書きと同じように確かめる。
  `./setup-deepseek.sh --model qwen38 --dflash` でダウンロードして設定に書く（`--no-dflash` で外す）。コードと一覧は MTP 層より
  速く、文章（とくに日本語）は遅いので、既定にはしていない:

  | 16G64G、RTX 5090（32 GB）、greedy 512 トークン、tok/s | コード | 一覧 | 日本語の文章 | swe 風の修正 |
  | --- | --- | --- | --- | --- |
  | ドラフターなし（n-gram だけ） | 91.0 | 93.6 | 93.6 | 99.9 |
  | MTP 層（`STRATA_Q4X_MTP`） | 140.2 | 137.2 | **105.7** | 128.2 |
  | DFlash 2（`STRATA_Q4X_DFLASH`） | **152.3** | **142.1** | 91.9 | 122.0 |

  12G32G（L448v2。カードを 12 GB に制限、RAM の枠 20 GB を 28 GB の cgroup の中で、CPU 8 スレッド）では、2 つのドラフターは
  ほぼ同じ: DFlash 2 がコード 58.0 / 一覧 72.6 / 日本語 46.8 / swe 37.9 tok/s、MTP 層が 58.5 / 72.6 / 47.9 / 36.7
  （ドラフターなしは 48.1 / 63.7 / 47.0 / 41.1）。
  GPU 1 枚のときだけ（層の分割では使わない）。`STRATA_Q4X_DFLASH_N`（5、1〜5）: 1 回の下書きの上限。
  `STRATA_Q4X_DFLASH_PMIN`（0.9）: 下書きに必要な selector の確信度。下回った最初の下書きで止める（ドラフター本来の既定の
  0.75 では一覧が 142 に対して 137。0.5 はどれも遅い）。下書きは numpy で書いた参照と合う（コードを教師強制で流して、1 回あたり
  3.83 個の下書きのうち 3.62 個が当たり。参照は 3.79 個のうち 3.59 個）。
  1 回の下書きは 1 つの CUDA グラフで動く（RTX 5090 で 1.14 ms、前は 1.96 ms）。`STRATA_Q4X_DFLASH_GRAPH=0` でカーネルを 1 つずつ
  起動する（下書きは同じ）。
  PCIe が詰まりやすいカードでは引き分け: RTX 4090 + 96 GB、UD-Q4_K_XL そのまま（greedy、長いコード生成 8K トークン）で、
  MTP 層が 55.6 tok/s、DFlash 2 が 56.4 tok/s。下書きの採用率は DFlash 2 が 86〜89%、MTP が 78% と差があっても、速さはほぼ
  同じでした（#39）。ここでは生成が PCIe 越しのエキスパート読み込みを待っている（`vram hit` 約 60%、1 トークンあたり RAM から
  約 20 回）ので、よいドラフターで 1 回の検証窓から出るトークンが増えても、窓ごとのエキスパート読み込みは変わりません。

## Ornith-1.5-122B

v1.3.46 から、同じエンジンが [Ornith-1.5-122B](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF)（Qwen3.5-MoE 系、
llama.cpp の `qwen35moe`）も動かします。gated delta-net 層 36 + full attention 層 12、1 層 256 エキスパート（8 個を
使用 + ゲート付きの共有エキスパート）、残差は普通の 1 本です。`./setup-deepseek.sh --model ornith`（v1.3.100）が zephel01 の IQ3_XXS-B（GPU 16 GB + RAM 64 GB 向け）を入れます。ほかの GGUF は手動でセットアップします。GGUF、pack の
コマンド、設定例、測定は [docs/ornith/README.md](ornith/README.md)（英語）に。要点:

- **メモリ。** unsloth の UD-Q4_K_XL は 79.7 GB: エキスパート 70.4 GB（256 × 48 層、3072 × 1024、IQ4_XS〜Q6_K）と、
  GPU に置く Q8_0 の dense 5.4 GB。32 GB のカードにはエキスパート約 25 GB が載り、残りは RAM 層 60 GB に入るので、
  94 GB の PC ではデコード中に SSD を読みません。
- **速度。** AI PRO R9700 + 8060S レーン: デコード 1 トークン 23〜25 ms（41〜43 tok/s。iGPU の代わりに CPU が
  レーンを計算すると 30 ms）、575 トークンのプロンプトはバッチ経路で 359 tok/s。llama.cpp との KL は 559 位置で 0.024（このモデルでは llama.cpp 自身の CPU と HIP の差が
  0.027〜0.029）。
- **投機デコード。** n-gram ドラフトが検証窓を通って動きます（既定は 1 窓 1 ドラフト、`STRATA_DS4_SPEC=<k>` で増やす）:
  コピー課題で 1 ドラフト 57 tok/s、3 ドラフト 76 tok/s。GGUF の NextN ブロックもドラフターになります（エキスパートを
  `tools/nextn_pack.py` で詰め直して `STRATA_GLM_MTP_GGUF` で読む）: コード 59 tok/s、散文 50、コピー課題 85
  （iGPU なしでは コード 36 → 42、散文 33 → 39）。
- **NVIDIA:** RTX 5090 で、IQ3_XXS-B は NextN ドラフターありで 116 tok/s（75% が採用、v1.3.100）。エキスパートの置き場を
  1.45 GB にしたとき（issue #23 の 8 GB のカード）は、プロンプトが 356 tok/s（2K）、デコードが 45.5 tok/s（v1.3.82）。
- **VRAM が少ないとき:** `--kv q4_0`（または `k8v4`、`int8`）で attention のキャッシュが fp16 の 28%（41%、53%）になり、
  残りはエキスパートの置き場に回ります（v1.3.134、#44）。Ornith-1.5-397B を 128K の文脈で 16 GB のカードでも起動できます。設定項目の `--kv` を参照。
- **まだないもの:** 会話スロット、レイヤー分割、画像。エキスパートカーネルはこの形状にまだ調整していない。

## Qwen3.8-27B（dense）

v1.3.138 から、エンジンは [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)（dense の Qwen3.5、llama.cpp の
`qwen35`）も動かします。gated delta-net 層 48 + full attention 層 16（query 24 ヘッド、KV 4 ヘッド）、全層が dense の
SwiGLU FFN（5120 → 17408）でエキスパートはなく、下書きは NextN ブロックがします。`./setup-deepseek.sh --model qwen38-27b`
が unsloth の UD-Q4_K_XL を入れて pack を作ります（`tools/iq_pack.py --gguf <ファイル> --out <フォルダ>/pack`、テンソルは
すべて元の形式のまま）。

- **メモリ。** 24 GB 以上の GPU ではファイル全体を VRAM に置きます（GPU 1 枚）。NextN ドラフターあり、F16 の KV、4K の
  プロンプトで、エンジンが使うのは文脈 8K で 19.0 GB、32K で 20.7 GB、64K で 22.8 GB、128K で約 27.7 GB（RTX 5090 で
  VRAM を確保して空きを減らして計測）。セットアップは 24 GB のカードには 32K（64K はカードにほかに何もなければ動いた）、
  32 GB には 128K を書き、約 21 GB 未満の GPU では止まります。空きが少ないとプロンプト経路のバッファが小さくなり（空き
  24 GB、文脈 32K / 64K でも 4K のプロンプトは約 2,500 tok/s、生成 130 tok/s）、それも入らないときは起動時か最初の
  プロンプトで止まります（"did not allocate"）。`--kv int8` で attention のキャッシュが半分になります。
- **12〜20 GB の GPU（v1.3.140）。** 各層の FFN の先頭の行を VRAM に、残りをピン留めした RAM に置きます。デコードと verify
  ウィンドウでは、GPU が自分の行を計算する間に CPU が残りを計算し（PCIe を通るのは隠れ状態だけ）、CPU の答えを足します。
  プロンプトの合間は、プロンプト経路の空いているバッファ（1〜1.6 GB）に層まるごとの FFN を貸し、プロンプトの開始時に返します。
  （v1.3.142 から、プロンプトのバッファが届いた層だけを入れ直します）。セットアップは RAM 32 GB を要求し、GPU に入る文脈の
  長さを書きます。エキスパートのない pack は、エンジンの GPU カーネルを最初に使うときに読み込みます（CUDA の lazy loading。
  起動時にすべて読み込むより VRAM が約 350 MB 少ない）。
- **8 GB の GPU（v1.3.142）。** FFN の行をすべて RAM に置いてもプロンプト経路の VRAM が足りないときは、head
  （`output.weight`、1 GB）の先頭の行だけを VRAM に置き、残りは CPU が計算します（GPU は自分の行を同時に計算。トークン、
  下書き、verify ウィンドウのすべて）。セットアップは 8〜11 GB の GPU に 8K を書き、9 GB 未満では「ほかのプログラムを GPU に
  載せない」よう警告します（余裕は約 0.2 GB）。`STRATA_Q35D_HEAD_ROWS=<n>` で分割を強制（テスト用）。
  RTX 5090 を制限、4K プロンプト + 512 トークン、3 回の中央値（llama.cpp: プロンプトは -ub 4096、生成は最良の -ngl）:

  | VRAM | 生成（NextN 下書きあり） | 生成（下書きなし） | プロンプト | llama.cpp 生成 / プロンプト |
  |---|---|---|---|---|
  | 8 GB | 15.5 | 8.4 | 642 | 8.0 / 1,121（-ub 既定値: 309） |
  | 12 GB | 28.4 | 14.1 | 1,525 | 13.9 / 1,525 |
  | 16 GB | 71.0 | 45.2 | 2,143 | 38.4 / 2,395 |
  | 20 GB | （すべて VRAM） | 72.4 | 3,100 | 72.4 / 3,585 |

  head を分割したときの llama.cpp との KL: 0.0025（トークン経路）/ 0.0020（プロンプト経路）/ 0.0034（verify ウィンドウ）。
  head をすべて VRAM に置く場合と同じです。
  `STRATA_Q35D_SPLIT=0`: 行ではなく層まるごとを CPU に置く（A/B 用。16 GB・下書きなしで 22.2、通常は 42.2）。
- **品質。** トークン ID は llama.cpp と同じ。llama.cpp との KL は英語・日本語・コード 3,071 位置で 0.0025（1 トークンずつ）、
  0.0019（プロンプト経路）。4K のチャットの greedy 生成は最初の 212 トークンが llama.cpp と一致。
- **速度。** RTX 5090、4K プロンプト + 512 トークン、3 回の中央値: デコードは NextN ドラフターありで 131.6 tok/s（下書きの
  82% が採用）、なしで 72.7。llama.cpp は 72.1。プロンプトは v1.3.141 で 2,538 tok/s（llama.cpp 3,463、-ub 4096 で 3,585）、
  v1.3.142 から 3,100 tok/s（下記）。
- **プロンプトの attention を tensor core で（v1.3.142、デフォルト）。** attention 層 16 層のプロンプトの attention を、KV ヘッド
  ごとの query 6 ヘッドのグループ単位で tensor core で計算します（FP16 の入力、F32 の和）。delta-net の scan は今の行を計算する
  間に次の行を読みます。ファイル全体を VRAM に置いた 4K プロンプトで 1,570 → 1,313 ms（attention 0.072 → 0.019 ms/トークン）。
  プロンプト経路の llama.cpp との KL は 0.00196（F32 カーネルは 0.00193）。プロンプト経路の出力は v1.3.141 とビット単位では
  一致しません。`STRATA_Q35_PATTN=1` で F32 カーネルに戻ります（v1.3.141 と同じ結果）。`=2` は MoE の qwen35 モデル（Ornith、
  Qwen3.8-Flash-Next。デフォルトではオフ）にも新しいカーネルを使います。
- **unsloth のリポジトリのすべてのファイル。** GPU のカーネル、プロンプト経路、CPU レーン（12〜20 GB / 8 GB の GPU）が、
  各ファイルの型をすべて読みます。Q4_0 / Q4_1 の行、BF16 の FFN の行（UD-Q8_K_XL、BF16）、プロンプト経路での i-quant
  （IQ1_S / IQ1_M / IQ2 / IQ3）と Q2_K の射影（MMQ がその型を持たない層は、層ごとに FP16 の積で計算）。RTX 5090 での実測（VRAM にすべて / 12 GB / 8 GB、下書きありの生成 tok/s）: UD-IQ2_XXS 114 / 117 / 81、UD-IQ3_XXS 141 / 140 / 27、UD-Q4_K_XL 133 / 30 / 17、UD-Q6_K 110 / 19 / 13、Q8_0 108 / 12 / 11。同じファイルでの llama.cpp との KL は 0.0005〜0.0025。CPU レーンの BF16 の行はまだ遅いです（12 GB 以下）。
- **別ファイルの NextN ブロック。** blk.64 のないファイルは、横の `MTP/` にあるもの（unsloth の
  `mtp-Qwen3.8-27B-Q4_0.gguf`）を使います。`STRATA_GLM_MTP_GGUF=<ファイル>` でほかのものを指定でき（自分のブロックを
  持つファイルでも）、`STRATA_GLM_NO_MTP=1` で下書きしません。別ファイルのブロックも自分のブロックと同じくらい当たります（UD-Q4_K_XL: 133.0 対 133.4 tok/s、採用率 82% 対 83%）。下書きありの greedy の答えは下書きなしと同じです（`STRATA_DETERMINISTIC=1`、同じチャンクの大きさ）。
- **DFlash 2 下書き役（v1.3.171）。** `STRATA_Q35_DFLASH=<ファイル>`（収まるときはセットアップが書く）で、z-lab の
  [Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) ブロック下書き役（対象の 6/20/34/48/62 層を
  読む 5 層、Q4_K_M 1.14 GB、VRAM 約 1.4 GB）が NextN ブロックと並んで、1 ステップに最大 7 トークンを出します。
  どちらで下書きするかはステップごとに実測の「トークン / ms」で決める（最初の 10 ステップは交互に試し、そのあとは
  ときどきもう一方を試す）。DFlash 2 の本数は確信度と実測の窓の時間から削り、コピー下書きは従来どおり最優先。
  答えが CJK の文章のあいだ（直近のピースの 3 分の 1 以上がかな・漢字）は NextN で下書きし、DFlash 2 は直近 256 行の
  文脈だけを持ちます。下書きはすべて検証するので、出力は下書きなしと同じ（deterministic md5 が一致）。
  RTX 5090、decode tok/s、3 回の中央値（ja / en / code / think: 128K コンテキスト、1,200 トークン。edit / swe-bench: 16K、512）:

  | ファイル | ja | en | code | think | edit | swe-bench |
  |---|---|---|---|---|---|---|
  | L-Q4_0、v1.3.170 | 121.2 | 128.8 | 185.6 | 135.1 | 292.0 | 309.3 |
  | L-Q4_0、DFlash 2 | 121.4 | 132.3 | **231.3** | **162.5** | 306.8 | 318.8 |
  | UD-Q4_K_XL、v1.3.170 | 100.6 | 109.0 | 158.4 | 128.0 | 227.1 | 153.1 |
  | UD-Q4_K_XL、DFlash 2 | 100.9 | 114.4 | **194.1** | **158.1** | 239.5 | **173.6** |

  1 回の検証で進むトークン数: code 3.29 → 4.90（L-Q4_0）、3.47 → 5.15（UD-Q4_K_XL）。VRAM を 24 GB に制限した場合
  （32K コンテキスト、512 トークン）: code 179.1 → 222.1 / 163.6 → 202.6、edit 290.9 → 307.1 / 229.0 → 240.5、日本語は同じ。
  L-Q4_0 の短い英語の答えは −0.9%（132.2 → 131.0、3 回の中央値。そこでは DFlash 2 と NextN の差が数 % しかなく、
  試すコストが出る）。ファイル全体と一緒に VRAM に
  載る必要があります。FFN が収まらないとき（16 GB のカード: FFN がさらに 1.3 GB RAM へ押し出され −22〜−37%）は、
  起動時にエンジンが外します（「dflash: dropped」。`STRATA_Q35_DFLASH_KEEP=1` で残す）。NVIDIA の GPU 1 枚のみ
  （層の分割では使わない。AMD は未検証）。`STRATA_Q35_DFLASH_PICK=1` / `2`: 常に DFlash 2 / 常に NextN（比較用）。
- **画像。** 設定に `vision` があれば（セットアップが書く）画像を読みます。OpenAI の `image_url`、Anthropic の `image`
  ブロックで渡します。画像の行はその 2 次元の格子の位置を取り（interleaved M-RoPE。区分は GGUF から読む）、画像のあとの
  文字はそこから続きます。画像 1 枚は 0.3〜0.6 秒ほど（225〜300 行）。 文字だけの要求は変わりません（前と同じトークン）。設定:
  [docs/SETTINGS.ja.md](SETTINGS.ja.md#qwen38-27b-の画像)。
- **まだないもの:** 複数 GPU へのレイヤー分割、AMD（コンパイルは通る。未検証）、AMD での画像。Qwen3.8-Flash-Next は画像を
  断ります。MoE の qwen35（Ornith）も同じ画像の経路を通りますが未検証です（変換器ファイルは試していない）。

## Ternary Bonsai 2.5 27B（ベータ）

エンジンは prism-ml の [Ternary-Bonsai-2.5-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF)（Apache-2.0）も
動かします。Qwen3.8-27B（dense の `qwen35`。[Qwen3.8-27B](#qwen38-27bdense) と同じ形）の重みを 3 値（-1 / 0 / 1、128 個に
FP16 のスケール 1 つ）にして、ブロックごとの Hadamard の基底で持つモデルです。2 つのファイルは同じ重みです:
`Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf`（6.23 GB、PTQ1_0 = 1 バイトに 3 値 5 個、MTP ブロックつき）と
`Ternary-Bonsai-2.5-27B-Q2_0.gguf`（7.67 GB、Q2_0、MTP ブロックなし）。`./setup-deepseek.sh --model bonsai27b` が PTQ1_0 の
ファイルを入れて pack を作ります（`--quant Q2_0` でもう一方）。文字だけです。

- **動かし方。** GGUF の `prism.hadamard.*` が、どの重みを回したかを書いています（401 本: すべての射影、FFN、head。1,024 個の
  ブロック、幅 5,120 / 6,144 / 17,408 の符号）。エンジンは積の前に活性化を回し（`x' = H (S . x)`。`ssm_out` は delta-net の
  出力の並べ替えも）、埋め込みの行は引いたあとに戻し、F32 の `ssm_alpha` / `ssm_beta` の行は読み込み時に一度回します。変換は
  FP32 で正確で（1/32 は 2 のべき）、積が読む q8_1 への量子化と一緒に 1 つのカーネルでします。トークン経路、プロンプト経路
  （PTQ1_0 / Q2_0 は自前の int8 GEMM）、検証窓、CPU レーン（RAM に置いた FFN の行）のすべてで回します。MTP ブロック
  （`blk.64`、Q4_0）は回していないので、そのまま NextN の経路で下書きします。
- **品質。** 2 つのファイルは同じ 3 値の重みになり（269 億個すべてと、スケール・F32 のテンソルを比べた。
  `recipes/bonsai27b/cmp_formats.py`）、トークン経路のロジットも同じです（英語・日本語・コードの 3,071 位置で KL 0.000000）。
  プロンプト経路とトークン経路: KL 0.00015、同じ最上位トークン 99.4%。同じ位置で Qwen3.8-27B の Q8_0 と比べると KL 0.405、
  同じ最上位 75.6%、nll 2.021（Q8_0 は 1.671。unsloth の UD-Q4_K_XL は KL 0.0067、96.4%）。モデルカードはタスクの成績が FP16 に
  近いとしています。元のモデルとの KL は、3 値に合わせて学習し直したモデルが元からどれだけ離れたかで、タスクの品質ではありません。
  英語・日本語・コード・ツール呼び出し・推論ありの貪欲の回答は正しいです。
- **速さ**（RTX 5090、iGPU なし、4K のプロンプト + 512 トークン、3 回の中央値。`bench/bonsai/speed.txt`）:

  | ファイル | 生成、MTP あり | 生成、下書きなし | プロンプト tok/s | VRAM 最大 |
  |---|---|---|---|---|
  | PTQ1_0 | **185.3**（採用 82%） | 125.5 | 3,162 | 11.8 GB |
  | Q2_0 | （MTP ブロックなし） | 130.6 | 3,313 | 13.0 GB |
  | Qwen3.8-27B UD-Q4_K_XL（比較用） | 137.4 | 76.8 | 3,105 | 21.5 GB |
  | PTQ1_0、上限 12 GB | 185.7 | | 3,154 | 8.8 GB |
  | PTQ1_0、上限 8 GB | 178.6 | 125.4 | 3,069 | 7.5 GB |
  | UD-Q4_K_XL、上限 12 / 8 GB | 28.0 / 15.6 | | 1,479 / 638 | |

  上限なしの VRAM 最大にはプロンプト経路のバッファが入っています（空きの VRAM から大きさを決める）。prism-ml のカード（自前の
  llama.cpp のフォーク、RTX 5090）: 下書きなし約 163〜174 tok/s、MTP 3 個で 305、2K のプロンプト約 4,200 tok/s。エンジンの
  3 値の GEMV はまだそれより遅く（トークン経路で 1 トークン約 8.3 ms。3 値はレジスタで復号）、速いカーネルが次の課題です。
- **メモリ。** ファイル全体を VRAM に置くときに文脈ごとに要る GPU の大きさ（PTQ1_0、MTP あり、F16 の KV、4K のプロンプト。
  RTX 5090 で上限をかけて計測）: 8K で 8 GB、32K で 9.5 GB、64K で 12 GB、128K で 16 GB（それぞれ生成 179〜187 tok/s。CPU に
  あるのは 1 層の最後の 1,024 行だけ）。セットアップは GPU に入るいちばん長い文脈を書きます（Q2_0 は +1.4 GB）。それより小さいと、
  多くの層の FFN の行を RAM に置いて CPU で計算します（CPU レーンは PTQ1_0 を読み、down の入力の自分の分を自分で回す）。
  動きますが遅く、9 GB で 32K は 30.5 tok/s、11 GB で 64K は 64.8 です。KV を必要なときに取る仕組みが、128K の使っていない分を
  文脈が伸びるまで FFN に貸します。長いプロンプト（上限なし）: 32K は 3,005 tok/s、そのあと生成 127.2。64K は 2,634、そのあと
  88.2（下書きなしで 74.5）。
- **まだないもの:** 画像（リポジトリに変換器のファイルがない）、AMD（コンパイルは通る。未検証）、複数 GPU への層の分割、DFlash 2
  下書き役（Qwen3.8-27B 用のもの。このモデルでは試していない）。Q2_0 のファイルは下書きしません（n-gram の下書きだけ）。

## Qwen3.5-9B QSA

自宅で学習した QSA indexer を足した Qwen3.5-9B です（indexer の形は higa2515 の Qwen3.8-27B-QSA と同じ。
[設定項目](#設定項目)の `STRATA_Q35_QSA` を参照）。ファイルは [zephel01/Qwen3.5-9B-QSA-GGUF](https://huggingface.co/zephel01/Qwen3.5-9B-QSA-GGUF)
の `Qwen3.5-9B-QSA-Q4_K_M.gguf` と `Qwen3.5-9B-QSA-Q8_0.gguf`（BETA）。v1.3.163 から
`./setup-deepseek.sh --model qwen35-9b-qsa` で入れられます（[まとめてセットアップ](#まとめてセットアップ)）。
full attention の 8 層のうち 3 層は dense のまま、ほかの 7 層は 4,099 セルを超えると indexer が選んだ上位 1,024 個の
4 セルブロックと自分のブロックだけに attention します。これはファイルのメタデータに書いてあります: 3 層の
`attention.compress_ratios` が 0（indexer テンソルもない）、`attention.indexer.top_k` が 4096。エンジンはこの 2 つを
読みます（v1.3.160 の次のコミットから。v1.3.157〜160 はこのファイルを全部 dense で動かす）。起動時の行の最後に
`dense attention layers: 3` と出ます。

```bash
.venv/bin/python tools/iq_pack.py --gguf Qwen3.5-9B-QSA-Q4_K_M.gguf --out pack
build-cuda/abyss --glm-pack pack --max-context 131072 --prefill auto
```

- **メモリ**（RTX 5090、MTP の下書きあり、ファイル全体を VRAM に置く）: GPU メモリの最大は文脈 32K / 64K / 128K で
  11.2 / 12.4 / 14.9 GB。そのうち 4 GB はプロンプト経路のバッファ（VRAM の空きが少ないと小さくなる）。ホストの RAM は 7.0 GB。11 GB に制限しても（`STRATA_GLM_VRAM_GB=11`）
  128K の文脈は動く: 最大 10.7 GB、FFN のうち 0.77 GB を RAM に置き、decode 245 tok/s。
- **速度**（RTX 5090、128K のチャットのプロンプト + 256 トークン、MTP の下書きあり、各 2 回。dense は同じファイルに
  `STRATA_Q35_QSA=0`）: decode 132 → 240 tok/s（1.81 倍）、プロンプト 6,018 → 5,546 tok/s（−8%）。
- **品質**（teacher-forced。学習に入れていない 128K のサンプル 8 本の最後の 1,024 トークン。同じファイルの dense と nll を比べる）:
  コード +0.6 / +1.0%、英語 −0.1 / +1.2%、日本語 +0.9 / +3.1%、エージェントの記録 +0.6 / +2.8%。平均 KL は agent を除いて
  0.032（含めて 0.042）。125K のニードル（秘密 3 つ + 関数の値、thinking あり）: 4/4。
- 制限は 27B の QSA ファイルと同じです: `--kv` と画像入りのリクエストでは dense。`STRATA_Q35_QSA=0` でファイルを dense で動かせます。
- AMD（HIP）: v1.3.170 から QSA が動きます（それまでは dense。issue #51 で RX 9070 XT の計測をいただきました）。Radeon 8060S
  （gfx1151、iGPU だけ）、128K のチャットプロンプト + 256 トークン: decode 30.3 → 48.1 tok/s（1.59 倍）、プロンプト 326 → 509〜585 tok/s。
  32K は decode +8%、プロンプト +11%。8K は同じ。5090 の QSA との KL は、dense 同士（5090 と 8060S）の KL と同じ大きさ。
  32K / 60K / 100K のニードル 4/4。
- このファイルは llama.cpp では読み込めません（`wrong number of tensors`。qwen35 の読み込みが indexer テンソルを知らないため）。

## Ornith-1.5-9B QSA

同じ QSA indexer を [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) に載せたものです（構造・形・トークナイザは
Qwen3.5-9B と同じ。3 層は dense、top_k 4096）。ファイルは [zephel01/Ornith-1.5-9B-QSA-GGUF](https://huggingface.co/zephel01/Ornith-1.5-9B-QSA-GGUF)
の `Ornith-1.5-9B-QSA-Q4_K_M.gguf`（5.9 GB、既定）と `Ornith-1.5-9B-QSA-Q8_0.gguf`（9.9 GB）（BETA）。v1.3.166 から
`./setup-deepseek.sh --model ornith-9b-qsa` で入れられます。GGUF には Ornith 自身のチャットテンプレートが入っています。文字だけで、
下書きはファイル自身の NextN ブロックです。[Qwen3.5-9B QSA](#qwen35-9b-qsa) に書いたことはそのまま当てはまります（同じエンジンの経路、同じ制限）。

- **メモリ**（RTX 5090、制限なし）: GPU メモリの最大は 32K / 64K / 128K で 11.2 / 12.4 / 14.9 GB。Qwen3.5-9B QSA と同じです。
  VRAM を制限した計測は Ornith では行っていません。形も最大値も同じなので、セットアップは Qwen3.5-9B QSA の表（11 GB の GPU、128K の文脈）を使います。
- **速度**（128K のプロンプト、Q4_K_M、MTP の下書きあり。dense は同じファイルに `STRATA_Q35_QSA=0`）: decode 138.1 → 214.8 tok/s
  （1.56 倍。MTP の採用率は dense 86.6% → QSA 82.8%、プロンプト 1 本）、プロンプト −7.9%。
- **品質**: 128K での dense との KL は 0.141（Q4_K_M。Qwen3.5-9B QSA は 0.032）。量子化だけの KL は 0.318 で Qwen の 0.072 に対し、
  比は同じ（0.44）。Ornith 自体が 128K で敏感なモデルです。125K のニードル: 4/4。
- このファイルは llama.cpp では読み込めません（indexer テンソルのため。Qwen3.5-9B QSA と同じ）。

## MiMo-V2.6-Distill-Qwen-9B

Xiaomi の [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)（Qwen3.5-9B の SFT、MIT。構造・形・
トークナイザは同じ）を、[bartowski の GGUF](https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF) から dense の qwen35 経路で動かします。
v1.3.168 から `./setup-deepseek.sh --model mimo-9b` で入れられます（既定は Q4_K_M、ほかに Q8_0。サンプリングは GGUF の値:
temperature 0.6、top_p 0.95、top_k 20）。文字だけです。

- **チャットテンプレート**: MiMo v2.6 のもの。プロンプトは `<|im_start|>assistant\n` で終わり、`<think>` はモデル自身が書きます。
  v1.3.168 から、サーバーはこの最初の `<think>` を reasoning の始まりとして扱います（それまでは `reasoning_content` の先頭に
  `<think>` の文字が入っていました）。ツール呼び出し（`<tool_call><function=...>`）は `tool_calls` で返り、ストリーミングでも
  返ります。`enable_thinking: false` も使えます。
- **下書き**: ファイルに NextN ブロックはありません。Qwen3.5-9B 自身のブロックが MiMo の下書きに使え、採用率は 76〜78%
  （Qwen3.5-9B 自身のファイルでは 83〜87%）。そのブロックだけを入れた GGUF を、モデルの GGUF の横の `MTP/` に置けば
  エンジンが自分で使います（セットアップがそこに置きます）。`STRATA_GLM_MTP_GGUF=<ファイル>` でも指定できます。
- **速度**（RTX 5090、Q4_K_M、チャットのプロンプト + 256 トークン、各 2 回。2 回とも同じ出力）:

  | | 4K 生成 | 32K 生成 | 4K / 32K プロンプト |
  |---|---|---|---|
  | MiMo、下書きブロックなし（n-gram の下書き） | 217 tok/s | 186 tok/s | 8,335 / 9,031 tok/s |
  | MiMo + Qwen3.5-9B のブロック | 273 tok/s | 228 tok/s | 7,914 / 8,688 tok/s |
  | Qwen3.5-9B Q4_K_M（自分のブロック） | 313 tok/s | 247 tok/s | 7,879 / 8,740 tok/s |

  128K のプロンプトでは、生成がブロックなし 117.5 tok/s、あり 126.0、プロンプトは 6,322 / 6,156 tok/s。
- **メモリ**（制限なし）: GPU メモリの最大は文脈 32K で 11.17 GB、128K で 14.27 GB（下書きブロックありで 15.11 GB）。ホストの RAM は 7.1 GB。
  Qwen3.5-9B と同じなので、セットアップは Qwen3.5-9B QSA の GPU の表（128K まで 11 GB。Qwen で制限して計測）を使います。

## MiMo-V2.6-Distill-Qwen-9B QSA

[MiMo-V2.6-Distill-Qwen-9B](#mimo-v26-distill-qwen-9b) に [Qwen3.5-9B QSA](#qwen35-9b-qsa) の indexer をそのまま足したものです
（BETA、zephel01/MiMo-V2.6-Distill-Qwen-9B-QSA-GGUF、Q4_K_M か Q8_0）。MiMo は Qwen3.5-9B の SFT で形が同じなので、Qwen3.5-9B で
学習した indexer がそのまま載ります（QSA は同じ 7 層、層 3 は dense、`top_k` 4096）。MiMo では学習していません。MiMo の重み・
キー・チャットテンプレートは bartowski のファイルのままで、足したのは indexer のテンソル 35 本と QSA のキーだけです。
v1.3.170 から `./setup-deepseek.sh --model mimo-9b-qsa` で入れられます。下書きブロックは `--model mimo-9b` と同じく借りたものを
`MTP/` に置きます。文字だけです。`STRATA_Q35_QSA=0` で同じファイルを dense で動かせます。

- **品質**（128K。Qwen3.5-9B QSA と同じ評価用の 8 本の最後の 1,024 トークンを teacher-forced で採点し、同じファイルの dense と比べる）:
  エージェントの 2 本を除いた KL の平均は Q4_K_M 0.124、Q8_0 0.105（2 本を含めると 0.234 / 0.213）。nll の差の平均は +0.2% / +0.6%。
  Qwen3.5-9B QSA は 0.032、Ornith で学習し直した Ornith-1.5-9B QSA は 0.141。dense より +2% を超えたのは、Q4_K_M の英語 2（+4.8%）と
  エージェント 3（+2.1%）、Q8_0 のエージェント 3（+2.5%）です。
- **125K のニードル**（埋め込んだタグ 3 つ + 関数の値、サーバー経由、thinking あり）: Q4_K_M は QSA・dense とも 4/4。Q8_0 は両方 3/4 で
  答えも同じ（関数の値を考えずに答えて外した）。タグの名前は `BUILD_TAG_*` です。`SECRET_*` だと MiMo は引用を断ります。
- **速さ**（RTX 5090、129,872 トークンのチャットのプロンプト + 256 トークン、借りた下書きブロックあり、各 2 回で同じ出力）:

  | | dense の生成 | QSA の生成 | プロンプト dense / QSA |
  |---|---|---|---|
  | Q4_K_M | 126.4 tok/s | **208.0 tok/s**（1.65 倍） | 6,155 / 5,665 tok/s |
  | Q8_0 | 124.2 tok/s | **180.7 tok/s**（1.45 倍） | 6,248 / 5,742 tok/s |

  32K（QSA）では 287.8 tok/s（Q4_K_M）、226.2（Q8_0）。indexer なしの MiMo は 227.1。
- **メモリ**（GPU のピークは nvidia-smi。制限は `STRATA_GLM_VRAM_GB`）: 制限なしで 32K / 128K が 11.67 / 15.36 GB（Q4_K_M）、
  14.65 / 18.34 GB（Q8_0）。Q4_K_M を制限すると、32K は 11 GB で全部 VRAM に載り 268 tok/s。128K は 11 GB では FFN の 1.32 GB が
  RAM に回って 145 tok/s に下がり、12 GB で 206 tok/s（ピーク 11.24 GB）、13 GB で全部 VRAM に載ります。Q8_0 の 128K は 15 GB で
  181 tok/s。Qwen3.5-9B QSA は 11 GB で 128K が同じ速さで動きますが、MiMo では借りた下書きブロックと検証の窓がその分を使います。
  そのためセットアップの表は 32K が 11 GB、128K が 12 GB です（Q8_0 は 3.7 GB 多く）。
- llama.cpp では読み込めません（`wrong number of tensors; expected 462, got 427`）。

## SenseNova-Loop-9B

SenseNova-Loop-9B（Apache-2.0）は、ループする Qwen3.5-9B です。32 ブロックを 56 層として回し（1 周目のすぐあとにブロック 4〜27 を
もう一度通る）、ループ部分の残差を 2 本の流れの mHC（サブ層ごとの小さなルーター: 入力ゲート 2 つ、出力ゲート 2 つ、Sinkhorn で
二重確率にした 2x2 の混合）で運びます。v1.3.169 から、[guideboardlabs の GGUF](https://huggingface.co/guideboardlabs/SenseNova-Loop-9B-GGUF)
（llama.cpp の arch は `sensenova_loop`）を dense の qwen35 経路で動かせます。`./setup-deepseek.sh --model sensenova-loop-9b` で入れられます
（既定は Q4_K_M、ほかに Q3_K_M / Q4_0 / Q8_0。サンプリングは GGUF の値: temperature 1.0、top_p 0.95）。文字だけです。

- **動き方**: 2 周目は 1 周目の重みを読みますが、KV の行と線形 attention の状態は自分のものを持ちます（56 層それぞれ別）。
  RoPE はファイルどおり YaRN（262,144 の 2 倍）。GPU 1 枚で動きます（層の分割はまだ）。FFN は VRAM に置きます。
  NextN ブロックはないので、n-gram の下書きを使います。
- **参照との比較**（モデル自身の llama.cpp fork、GuideboardLabs/llama.cpp-sensenova-loop、CUDA。教師強制の log-softmax で
  KL(fork || Abyss)）:

  | | en | ja | zh | code | 5K の code（プロンプト経路） |
  |---|---|---|---|---|---|
  | Q4_K_M | 0.0091 | 0.0093 | 0.0060 | 0.0050 | 0.0080 |
  | Q3_K_M / Q4_0 / Q8_0 | 0.0027 / 0.0012 / 0.0010 | 0.0042 / 0.0013 / 0.0011 | | 0.0011 / 0.0002 / 0.0004 | 0.0031 / 0.0015 / 0.0017 |

  Q4_K_M の値は、このエンジンがどの Q4_K_M ファイルでも出す水準です（plain な Qwen3.5-9B Q4_K_M と llama.cpp の比較で 0.0085）。
  6K の日本語文書では 0.131 で、ほぼすべてが数字の表の中の 10 か所ほどから来ます。同じ所で fork 自身の CPU と CUDA を比べると
  0.265 です（位置ごとの中央値は 0.0014、Abyss は 0.0012）。
- **速度**（RTX 5090、チャットのプロンプト + 256 トークン greedy、n-gram の下書き。各 2 回、同じ出力）:

  | | 4K 生成 | 32K 生成 | 128K 生成 | 4K / 32K / 128K プロンプト |
  |---|---|---|---|---|
  | Q4_K_M | 102.3 tok/s | 96.6 tok/s | 62.6 tok/s | 5,120 / 5,045 / 3,553 tok/s |
  | Q4_K_M、下書きなし | 100.1 | 90.1 | | |
  | Q4_0 | 120.3 | 101.1 | | 5,525 / 5,542 |
  | Q3_K_M | 109.3 | 98.1 | | 4,604 / 4,545 |
  | Q8_0 | 80.1 | 75.4 | | 5,285 / 5,138 |

  Qwen3.5-9B の 32 層に対して 56 層あるので、速さは Qwen3.5-9B のおよそ半分です。
- **メモリ**（Q4_K_M）: 制限なしでは GPU メモリの最大が 8K / 32K / 64K / 128K で 10.2 / 12.0 / 13.8 / 17.3 GB。増える分の大半は
  プロンプト経路のバッファで、エンジンは空いている VRAM から大きさを決めます。`STRATA_GLM_VRAM_GB` で 10 / 12 / 14 GB に制限すると、
  32K / 64K / 128K で 8.4 / 10.2 / 13.8 GB になり、生成の速さは同じです（プロンプトは最大 8% 遅くなります）。ホストの RAM は 6.9 GB。
- **日本語**: 8 問（説明、ビジネスメール、コード、助言、歴史、算数、翻訳、物語）はすべて日本語で答えます。考える部分はおもに英語です。
  一部の回答に中国語の語が少し混じります（气也没有、量子比特、持续、最后一次。8 問中 3 問。`STRATA_LANG=ja-en` では 2 問で、
  日本語でも使う文字は文字の種類で絞る設定では止められません）。英単語（"consideration"）も 1 つ混じりました。上の KL から、
  これは移植ではなくモデル自身の癖です。
- 決定的モード（`STRATA_DETERMINISTIC=1`）では、何度動かしても同じ出力になります。

## Laguna-S-2.1

v1.3.140 から、エンジンは poolside の [Laguna-S-2.1](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)（`laguna`、118B の
MoE、1 トークンあたり約 8B が動く）を qwen35moe の層経路で動かします。`./setup-deepseek.sh --model laguna --download-model`
が unsloth の UD-IQ4_XS（58 GB）を入れます。RTX 5090 を 16 / 24 / 32 GB に制限: 生成 v1.3.142 から 42.5 / 50.9 / 60.6 tok/s（v1.3.140 では
36.3 / 41.4 / 47.4。llama.cpp 32.0 / 38.1 / 43.2）、プロンプト 859 / 1,005 / 1,174（318 / 391 / 470）、llama.cpp との KL 0.024
（llama.cpp 自身の CPU と CUDA の差と同じ）。NextN ドラフターはなく、v1.3.142 から n-gram の下書きが有効です。詳しくは [docs/laguna](laguna/README.ja.md)。

## MemT-1.0-Nano（ネイティブのメモリ、ベータ）

v1.3.204 から、エンジンは MemTensor の [MemT-1.0-Nano](https://huggingface.co/MemTensor/MemT-1.0-Nano)（Apache-2.0）を、
そのモデル自身のメモリつきで動かします（NVIDIA）。骨格は Qwen3.6-35B-A3B（qwen35moe の経路。形は Ornith-1.5-35B-A3B と同じ）で、
10 層あるフルアテンションの層それぞれにメモリのブロックがあります。メモリは 1 層あたり 16 x 256 x 256 個の FP32 の状態
（全部で 40 MiB）です。モデルは答えるあいだこれを読み、ターンの終わりにそのターンを書き込みます（ターンをモデルに流し直す）。
1 回話した事実を、それを含まない新しい会話で聞けます。setup の選択肢と公開の GGUF はまだありません。変換・pack・設定は手で行います。

変換（`tools/memt_convert.py`。Abyss 自前の読み書きで、llama.cpp・safetensors・torch は使いません）。`--meta` は、キーとトークナイザーを写す
同じ系統の GGUF で、Ornith-1.5-35B-A3B の Q4_K_M を使います（`./setup-deepseek.sh --model ornith35` がダウンロードします）。
メモリのブロックは BF16 / F32 のままです（0.9 GB）。

```bash
# Q8_0（37.8 GB）: numpy だけ
.venv/bin/python tools/memt_convert.py <MemT-1.0-Nano のフォルダ> MemT-1.0-Nano-Q8_0.gguf --meta=Ornith-1.5-35B-Q4_K_M.gguf
# Q4_K_M（22.1 GB）: テンソルごとの型は Ornith の Q4_K_M と同じ。k-quant は trench-gguf の GPU 量子化で
.venv/bin/python tools/memt_convert.py <MemT-1.0-Nano のフォルダ> MemT-1.0-Nano-Q4_K_M.gguf --meta=Ornith-1.5-35B-Q4_K_M.gguf \
  --types-from=Ornith-1.5-35B-Q4_K_M.gguf --gpuq=<trench-gguf>/gpuq/libgpuq_cuda.so
.venv/bin/python tools/iq_pack.py --gguf MemT-1.0-Nano-Q4_K_M.gguf --out <dir>/pack
```

設定: `"args": ["--glm-pack", "<dir>/pack", "--max-context", "16384", "--prefill", "auto"]`、`"tokenizer": "<dir>/pack/tokenizer"`。
サーバーは `--engine strata` で起動します。`memory` のない要求は骨格だけで答えます（Qwen3.6-35B-A3B と同じ答え方）。

サーバーのメモリの API（メモリはファイル `STRATA_MEMT_DIR/<id>.safetensors`、既定は `~/.abyss/memt`。id は英数字と `.` `_` `-` で 1〜128 文字）:

| 要求 | すること |
| --- | --- |
| チャットの要求（OpenAI の `/v1/chat/completions`、Anthropic の `/v1/messages`）に `"memory": {"id": "<id>", "commit": true}`（Anthropic は `metadata.memory` も可。`"memory": "<id>"` は読むだけ） | 答えはそのメモリを読みます（ファイルがまだなければ空のメモリ）。`"commit": true` なら、答えのあと（finish が `stop` か `length`）、要求の最後のユーザー発言と答えをメモリに書き、ファイルに保存します。返事に `"memory": {"id", "committed_tokens", "updated"}` が付きます。書いたターンは、次から送らなくてかまいません |
| `POST /v1/memory/<id>/documents` `{"text": "..."}` | 文章を文書としてメモリに書き（2,048 トークンごとの区切り）、保存します |
| `GET /v1/memory` | 保存したメモリの一覧（id、バイト数、更新時刻） |
| `POST /v1/memory/<id>/delete` | ファイルを消します（読むのもやめます） |

ファイルは MemT の `save_state` と同じ並びです（`memory.<層>` F32 [1, 16, 256, 256]、`format` = `memt-memory-v1`）。HF のコードで保存した
状態をここで読めますし、その逆もできます（identity のハッシュは確かめません）。エンジンのコマンド（`--serve`）:
`MEMT OFF|ZERO|INFO|LOAD <file>|SAVE <file>|COMMIT <ids>|DOC <ids>`。

RTX 5090（iGPU レーンなし）で測りました。基準は HF のコードから書いた float64 の numpy の順伝播（`recipes/memt/ref_memt.py`）、KL(基準 || エンジン):

| 確認 | 結果 |
| --- | --- |
| 骨格、メモリなし、Q8_0 | 49 トークンのチャット: KL 0.0014（トークン経路）/ 0.0078（プロンプト経路）、上位 1 位の一致 97.9% / 100%。長い文書の先頭 512 トークン: 0.052 / 0.034（64 位置ごとの中央値 0.001〜0.004。平均を押し上げているのは、基準自身も迷う目次の数か所）。同じ 512 で Q4_K_M: 0.194（プロンプト経路）、一致 88.3% |
| メモリの読み、Q8_0、基準が書いた状態を読み込む | KL 0.003（トークン経路、最大 0.013）/ 0.004（プロンプト経路）、28 位置すべてで上位 1 位が一致。同じ入力をメモリなしで流すと KL 3.01、一致 37% |
| メモリの書き、Q8_0、空の状態に 58 トークン | 層ごとの相対誤差 0.8〜2.8%（プロンプト経路）/ 1.4〜3.6%（トークン経路）、cos 0.9994 以上。深い層ほど大きい（Q8_0 の骨格の誤差が層の入力に入るため） |
| 思い出し（MemT の README の子猫の例） | 1 ターン目を書いたあと、質問だけの新しい会話で 4 点（名前・年齢・好物・特技）すべて正解。Q8_0 の CLI とサーバー、Q4_K_M のサーバーで確認。空のメモリでは「分からない」と答える。`/documents` で書いた 62 トークンのメモ: 2 問とも正解 |
| 速さ、Q4_K_M、4K のプロンプト＋512 トークン、3 回の中央値 | 生成はメモリなし 304.8 tok/s、あり 279.8（−8.2%）。プロンプト 8,802 → 8,336 tok/s（−5.3%）（v1.3.205。v1.3.204 のあり 295.5 はメモリを読まない検証の窓を通る下書きありの値） |
| ターンを書く時間 | 194 トークンで 0.17 秒（Q4_K_M、サーバー）。58 トークンで 0.6 秒（Q8_0、最初の書き込み。プロンプト経路の準備を含む） |

ベータです。NVIDIA だけ（カーネルは HIP でもビルドできますが、動かしていません）。モデルに入るメモリは一度に 1 つ（別のメモリの要求が来ると、
そのファイルを読み込みます）。MemT では会話スロット・`--parallel`・層の分割・下書きは使いません（MTP の重みがありません）。文書の書き方
（ユーザー発言 1 通として描画し、生成プロンプトを付けない）は MemT 自身のコードとは照らしていません（その文書への答えは正しく出ました）。
骨格が Qwen3.6-35B-A3B とバイト単位で同じかは確かめていません（元の重みが手元にないため）。

## MemT-1.0-Flash（ネイティブのメモリ、ベータ）

v1.3.205 から、エンジンは MemTensor の [MemT-1.0-Flash](https://huggingface.co/MemTensor/MemT-1.0-Flash)（MIT）を、そのモデル自身のメモリつきで
deepseek4 の経路で動かします（NVIDIA）。MemT-1.0-Flash は DeepSeek-V4-Flash-0731 を追加学習したもので、20 層の CSA 層（4, 6, ..., 42）それぞれに
メモリのブロックがあります。メモリは 1 層あたり 64 x 512 x 512 個の FP32（全部で 1.25 GiB、ファイルは 1.34 GB）です。サーバーのメモリの API は
[MemT-1.0-Nano](#memt-10-nanoネイティブのメモリベータ) と同じです。setup の選択肢と公開の GGUF はありません。ファイルは、私たちの
DeepSeek V4 Flash 0731 の `32G96G` GGUF（L97）の上に手で作ります。

MemT のエキスパートは 0731 のものと同じです（0731 の MXFP4 の値にそのまま戻ります）。要るのはメモリのブロックと MemT 自身のエキスパート以外の骨格だけで、
576 GB のチェックポイントのうち 21.9 GB です:

| もの | 取り先 | 大きさ |
| --- | --- | ---: |
| メモリのブロック（`memt-0000N-of-00002.safetensors`）と `config.json` | [MemTensor/MemT-1.0-Flash](https://huggingface.co/MemTensor/MemT-1.0-Flash)、ファイルごと | 7.2 GB |
| MemT のエキスパート以外の骨格（注意、共有エキスパート、ルーター、hyper-connection、埋め込み、ヘッド） | 同じリポジトリの 43 個のモデルの分割ファイルから、HTTP の範囲読み（44 回） | 14.7 GB |
| 元の GGUF | [zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4-Flash-0731-32G96G-GGUF) の `L97` フォルダ | 96.8 GB |
| trench-gguf の GPU 量子化（`gpuq/libgpuq_cuda.so`、v0.5.0 以降） | [zephel01/trench-gguf](https://github.com/zephel01/trench-gguf) | - |

```bash
# 1. メモリのブロックとエキスパート以外のバイト（途中から再開できます。HF のトークンがあれば HF_TOKEN か ~/.cache/huggingface/token）
.venv/bin/python tools/memt_flash_fetch.py <dir>/MemT-1.0-Flash-src
# 2. 重ねる GGUF: L97 のキー、MemT の骨格を L97 と同じテンソルの型で（k-quant は GPU で）、メモリの重みは Q8_0。
#    分割 2〜4 は L97 へのリンク（書くのは 9.3 GB、RTX 5090 で約 20 秒）
.venv/bin/python tools/memt_flash.py <dir>/MemT-1.0-Flash-src <L97>/DeepSeek-V4-Flash-0731-L97-00001-of-00003.gguf \
  <dir>/MemT-1.0-Flash-B --outtype=q8_0 --backbone=<dir>/MemT-1.0-Flash-src --gpuq=<trench-gguf>/gpuq/libgpuq_cuda.so
# 3. pack
.venv/bin/python tools/iq_pack.py --gguf <dir>/MemT-1.0-Flash-B/overlay-00001-of-00004.gguf --out <dir>/MemT-1.0-Flash-B/pack \
  --compat-bf16 --first-wins
```

imatrix は要りません。上のコマンドが測った B です。`--imatrix=<imatrix.gguf>` で骨格の k-quant に重みを付けられますが、私たちの 0731 の
imatrix（公開していません）で作っても、量子化そのものの揺れの範囲で同じでした（メモリなし、下の 5 本: NLL -0.009〜+0.006、
KL(B || imatrix あり) 0.009〜0.031、上位 1 位の一致 91〜96%。メモリあり: KL 0.022〜0.048。思い出しはどちらも 19 問中 19）。`--backbone` なしでは L97（0731）の骨格のままメモリだけを足します（「A」、書くのは 3.8 GB）。
MemT 自身の骨格（「B」）のほうが思い出しが正確で NLL も低い（下の表）ので、B を使ってください。

設定: `"args": ["--glm-pack", "<dir>/MemT-1.0-Flash-B/pack", "--max-context", "16384", "--prefill", "auto"]`、
`"tokenizer": "<dir>/MemT-1.0-Flash-B/pack/tokenizer"`。サーバーは `--engine strata` で起動します。`memory` のない要求は MemT の骨格だけで答えます。

かかるもの:

- メモリなし（要求に `memory` がない）: なし。メモリの重みは pack のマッピングに置いたままで、状態も取らないので、エキスパートの置き場は
  L97 の pack そのままと同じです（RTX 5090 で 23.97 GB）。
- メモリあり: 読み用の重み（Q8_0 で 1.2 GB）と BF16 に丸めた状態（0.67 GB）を、エキスパートの置き場の各層の末尾 7 枠に移します（置き場の 2.33 GB）。
  状態の FP32 の本体は RAM に置きます（1.34 GB）。メモリを切り替えると 1.34 GB のファイルを読みます。
- ターンや文書を書くとき: その流し直しの間だけ、書き用の重み（2.8 GB）と FP32 の書きの状態（1.34 GB）を置き場の尾の終わりに置き（4.1 GB、終われば返す）、
  ファイルを書き直します（ディスクにはメモリ 1 つで 1.34 GB）。

速度（RTX 5090、iGPU の経路なし、123 GB の機械で RAM の段は既定、学習したプールのファイルあり。4K のプロンプト＋512 トークン、
1 回温めてから 3 回の中央値。`bench/MEASURE.md` の手順）:

| | decode tok/s | prompt tok/s |
| --- | ---: | ---: |
| L97（DeepSeek V4 Flash 0731 そのまま） | 78.9 | 988 |
| MemT-1.0-Flash B、メモリなし | 77.5 | 983 |
| MemT-1.0-Flash B、メモリあり（325 トークン書いたメモリ） | 68.2（なしより -12%） | 867（-12%） |

メモリありで遅くなるのは、ほとんどが読みの計算そのものです（置き場の VRAM の当たりは、なし 81% に対して 77〜79%）。325 トークンのターンを書くのは
`COMMIT` 3.3〜3.5 秒（うち流し直し 1.8〜1.9 秒）、そのあと保存 0.7〜2.7 秒。メモリの切り替え（`LOAD`）は 1.2〜1.4 秒です。

RTX 5090 で、HF のコードから書いた float64 の numpy の計算（`recipes/memt-flash/ref_block.py`）と、エンジン自身の入力で比べました（層 4 / 22 / 42）:

| 確かめたこと | 結果 |
| --- | --- |
| メモリの書き（370 トークン）、同じ Q8_0 の重みの計算と | 相対誤差 4.2e-4 / 3.3e-4 / 4.5e-4。MemT の BF16 の重みとは 1.9% / 2.0% / 1.7%（cos >= 0.9998） |
| メモリの読み、同じ Q8_0 の重みの計算と | プロンプト経路 1.1% / 0.73% / 0.49%、トークン経路 0.60% / 0.98% / 0.56%（参照側の BF16 の丸めだけで 0.34〜0.36%） |
| 空のメモリ（`MEMT ZERO`）とメモリなし | 同じトークン |
| 思い出し（MemT の README の子猫の例） | ターン 1 を書いて、質問だけの新しい会話で 4 項目とも正しい。メモリなしでは「知らない」 |
| 文書として書いた日本語のメモ（83 トークン） | 5 項目（稼働日、場所、責任者、目標、検査）とも正しい。メモリなしでは知らない |
| B と A、メモリなし、1,024 トークン × 5 本（英語、コード、日本語） | 5 本とも B の NLL が低い（例：日本語 3.012 → 2.877）。KL(A \|\| B) 0.023〜0.061 |
| 思い出しの 19 問（子猫 4、日本語のメモ 5、1 ターンで伝えた架空の事実 10 を 1 つずつ、それぞれ新しい会話で。`recipes/memt-flash/memeval.py`） | メモリなし 19 問中 0、同じ会話の前に事実があるとき 19、メモリから 19（MemT-1.0-Nano の Q4_K_M と Q8_0 も 0 / 19 / 19） |
| 関係のない文をメモリありで読む（同じ 5 本、1 ターンのメモリかメモのメモリ） | NLL +0.18〜+0.49（Nano は +0.19〜+0.83）、KL(なし \|\| あり) 0.26〜0.46。メモリは書いた会話のためのもの。ほかの用途では切ってください（要求に `memory` を付けない） |

ベータです。NVIDIA だけ。モデルに入るメモリは一度に 1 つ。MemT では DSpark と MTP の下書きは使わず、n-gram の繰り返しの下書きはメモリを使っていないときだけです
（検証の窓がメモリを読まないため）。会話スロット、`--parallel`、層の分割も使いません。書くにはエキスパートの置き場の尾が要ります（32 GB の GPU で 13 GB あります。足りないと、理由を言って書くのを断ります）。

## Hy4-flash（ベータ、未リリース）

エンジンは Tencent の Hy4-flash（GGUF のアーキテクチャ `hy_v4_flash`。シンク付きのゲート MLA 注意、4 本の hyper-connection、
窓の層と疎な索引の層、潜在空間の MoE、SSD から読む Engram の n-gram 表）を動かします。試したのは
[peasantsmith/Hy4-Flash-Maya](https://huggingface.co/peasantsmith/Hy4-Flash-Maya) の `Maya-XS`（エキスパート IQ2_XXS、Engram FP8）です。
読み込みは、ほかの変換ツールのファイル（Engram 表が IQ4_NL / Q8_0 / F16 / BF16 / F32 でどの分割ファイルにあってもよい、
射影が BF16 / F16）にも対応しますが、そのようなファイルではまだ試していません。setup にはまだ入っていないので、手でパックして設定を書きます。

```bash
python tools/iq_pack.py --gguf <dir>/Hy4-Flash-Maya-XS-IQ2_XXS-00001-of-00002.gguf --out <dir>/pack-hy4
```

設定は `"args": ["--glm-pack", "<dir>/pack-hy4", "--max-context", "16384"]`、`"tokenizer": "<dir>/pack-hy4/tokenizer"` とし、
サーバーは `--engine strata` で起動します。思考は `reasoning_effort` に従います（`none` で切る。テンプレートにあれば `max`）。
RTX 5090（Gen4 x4）+ RAM 段 60 GB、iGPU レーンなし、459 トークンのプロンプト、512 トークン生成の 3 回の中央値:
生成 60.0 tok/s。32 トークン以上のプロンプトは、まとめて読む経路（未リリース）を通ります。注意・Engram の行（チャンク分を
まとめて読む）・射影をチャンク単位で計算し、潜在空間のエキスパートは共通のエキスパートの仕組みで計算します。同じ機械・
コードのプロンプト・その後 256 トークン生成で、プロンプト tok/s（1 トークンずつ → まとめて）: 459 トークン 62 → 487（3 回の中央値）、
4,096 トークン 49 → 1,092、15,808 トークン 50 → 1,573。その後の生成は変わりません（61 / 49 / 48）。この経路があれば
`--max-context 65536` も使えます。秘密 3 つ + 関数 1 つを埋めたプロンプト（bench/ctx128 のクライアント）を 32K と 63K
トークンで試し、どちらも 4 問すべて正解、プロンプト 1,351 / 1,381 tok/s、生成 48 / 51 tok/s（65,536 でキャッシュ 0.95 GB）。
設定: `STRATA_HY4_PREFILL=0` でプロンプトを 1 トークンずつに戻します（v1.3.177 と同じトークン経路）。
`STRATA_HY4_PREFILL_SUB=<64-4096>` は注意と密な FFN のサブバッチ（512）、`STRATA_HY4_PREFILL_READERS=<n>` は
チャンク分の Engram の行を読むスレッド数（32）です。`STRATA_HY4_PREFILL_ATTN=f32` でプロンプトの注意を F32 のカーネルに戻します
（既定は tensor core、sm_80 以上）。`STRATA_HY4_IX_TILED=0` でインデクサのスコアをワープ 1 本でグループ 1 つの版に戻します
（既定はレジスタでタイルを組む版、16 ヘッド）。`STRATA_HY4_ENG_ASYNC=0` で、トークンの Engram の行を層の計算の前に読み終える
形に戻します（既定は層 0 の計算中に読み、各 Engram 層が自分の行だけを待つ。値は同じ）。
`STRATA_HY4_PREFILL_ENG_ASYNC=0` でプロンプトのチャンクの Engram の行を、その層の計算の前にまとめて読む形に戻します（既定は層の計算中に読み、各 Engram 層が自分の行だけを待つ。同じ行）。Engram の gate の 3 つの norm（value、key、gate を掛けるストリーム）には FP32 の machine epsilon（1.19e-7）を足します。モデルの参照コードと同じです（参照コードの `RMSNorm(eps=None)` はBF16 の入力にもこの値を足します）。v1.3.198 までは bf16 の 2^-7 を足していて、層 1 と 11 の小さいストリームが縮み、gate が 2〜8 倍開いていました。BF16 の重みでの参照計算と比べた KL（混在 457 トークン、ja / en / code 各 1,024）は、Q8_0 で 1.21 / 0.46 / 1.38 / 0.36 → 0.12 / 0.11 / 0.04 / 0.03、自作の 32G96G pack C で 1.25 / 0.71 / 1.47 / 0.31 → 0.57 / 0.51 / 0.74 / 0.21。JSON / コードの崩れた答えは 2/48 → 0/48 です。`STRATA_HY4_ENG_EPS=0.0078125` で前の値に戻せます。`STRATA_GLM_PREFILL_PROF=1` /
`STRATA_GLM_PROF=<n>` でプロンプト / 生成の時間を段階ごとに出します（`hy4_*` とトークン経路の各段階）。画像: mmproj
（`mmproj-Hy4-Flash-Maya-XS.gguf`）は strata-vision で動きます（`tools/vision/hy4v_patch.py`、setup の vision ビルドが当てます）。
`"model"` には `python tools/vision/hy4v_ref.py vocab --gguf <1 本目の分割ファイル> --out <dir>/hy4-vocab.gguf` で作る語彙だけの GGUF を指定します。
まだのもの: MTP ドラフター、AMD（HIP）。

## EXL3 量子化（GLM-5.3-Flash）

v1.3.35 から、エンジンは [exllamav3](https://github.com/turboderp-org/exllamav3) の EXL3 量子化の GLM-5.3-Flash も、
safetensors からそのまま読む。詳しくは [docs/exl3/README.md](exl3/README.md)（英語）。要点:

- **インストール。** `./abyss.sh --setup` のモデル選択に、Maya の GGUF と並んで turboderp の `EXL3-sc_2.10bpw`（88.3 GB）と
  `EXL3-3.05bpw`（125.3 GB）が出る。それぞれにこの PC での向き不向きを表示する: RTX 30 以降のカードに 8 割以上載るなら
  おすすめ、APU の iGPU か AVX-512 の CPU が RAM 層を計算できるなら品質優先の選択肢、AVX2 だけの CPU には向かない。
  `--model EXL3-sc_2.10bpw` で質問を飛ばせる。セットアップでは NVIDIA の GPU のみ。
- **切り替え。** 量子化ごとに設定ファイルと起動スクリプトができる（`maya-exl3-sc_2.10bpw.json`、
  `run-maya-exl3-sc_2.10bpw.sh`）。GGUF と EXL3 を両方入れておき、起動するスクリプトで選ぶ。エンジンは
  `pack/exl3.txt` があれば EXL3 として読むので、設定項目は要らない。
- **手作業で。** `.venv/bin/python tools/exl3_pack.py --model <dir>`（約 5 秒）で `<dir>/maya-exl3.gguf` と `<dir>/pack/`
  ができる。起動は `--glm-pack <dir>/pack`。
- **目安**（RTX 5090 + Radeon 8060S の lane + RAM 94 GB、採点 1,000 トークン）: sc_2.10bpw は nll 1.591〜1.598、1 位一致
  63.3〜63.7%（Maya-S-v2-IQ2_XXS は 1.634 / 62.3%）。1 トークン 18.8 ms（同 15.4 ms）。v1.3.36 で iGPU のトレリス
  復号を `v_sad_u8` と `v_dot2_f32_f16` にした（v1.3.35 は 22.4 ms）。

## 複数のリクエストを同時に（`"parallel"`、Inkling 2 Mini、ベータ）

既定ではサーバーはリクエストを 1 本ずつ答え、残りは待たせる。Inkling 2 Mini（全 expert が VRAM）では、エンジンが
複数をまとめて動かせる: 実行設定に `"parallel": 4`（または `args` に `--parallel 4`。`"auto"` は全 expert が VRAM に
残る範囲で最大 8。それより大きい N はそこまで減らす: expert が VRAM の外に出ると答えが変わるため）。リクエストごとに
自分の lane（KV キャッシュと短い畳み込みの状態、`--max-context` 16K トークンあたり約 0.97 GB）を起動時に確保するので、
混んできてから VRAM が足りなくなることはない。N 本を超えたリクエストは待ち行列（先着順）に入る。32 GB のカードで
16K トークンなら 8 本、40960 なら 4 本。短いプロンプトは長いプロンプトの残りより先に読むので、短い質問が他の
リクエストの 32K の文脈の後ろで待たされることはない。

目安（RTX 5090、UD-Q4_K_XL、greedy、512 トークン。Abyss-tools の `bench/batch/drive.py`）:

| 動いているもの | 1 本あたり tok/s | 最初の 1 トークンまで | トークン間の最長の間 |
| --- | ---: | ---: | ---: |
| 1 本だけ | 196 | 0.04-0.06 s | 9 ms |
| 同時 2 本 | 128-129 | 0.05-0.11 s | 12 ms |
| 同時 4 本 | 90-92 | 0.01-0.20 s | 15-17 ms |
| 同時 8 本 | 59-62 | 0.01-0.49 s | 21 ms |
| 1 秒ごとに 1 本、計 4 本 | 106-131 | 0.01-0.04 s | 12 ms |
| チャット 2 本（2 本目は 0.5 s 後） | 139 ずつ | 0.01-0.03 s | 8 ms |
| 1,024 トークンの長い答えの間に短い質問 4 本が出入り | 154（長い方）、127-129 | 0.01-0.03 s | 8-10 ms |
| 3 本が動いているところに 8K トークンのプロンプト | 83-84（3 本） | 1.45 s（8K の方。単独 1.03 s） | 37 ms（3 本） |
| エージェントの 8K 文脈（lane が保持）、0.3 s 後に別の 8K、短い質問 | 99 / 122 / 88 | 0.15 / 1.29 / 0.23 s | 35 ms |
| 同じ 8K の会話の次のターン（lane が会話を保持） | 190 | 0.18 s | 8 ms |
| 3 本中 1 本を 50 トークンでキャンセル | 125-127（残り 2 本） | - | 15 ms |
| 40960 トークン・4 本: 16K の文脈、0.5 s 後に 32K、1 s 後に短い質問 | 61 / 124 / 29 | 2.5 / 7.1 / 1.5 s | 38 ms（32K を読んでいる間） |

全リクエストの合計: 同時 2 本 251 tok/s、4 本 342、8 本 450（1 本だけ 196）。

- **答えは 1 本のときと同じ。** まとめたステップの各行は、そのリクエストが単独で計算するものと同じ計算（自分の位置、
  自分のキャッシュ、同じ順の和）なので、同時に流した答えは単独のときとビット単位で同じ（上の全リクエストで md5 を確認）。
  プロンプトがプロンプト経路の chunk（32 GB のカードで 1,280 トークン）以下なら `"parallel": 1` とも同じ。それより長い
  プロンプトは、ほかに何が動いていてもその chunk ごとに読む。`"parallel": 1` では 32K を 1 回で読むので、そこでは
  答えの最後のビットが違うことがある（`--prefill` の chunk を変えたときと同じ）。1 本だけのときは通常の token の経路で decode する。
- **新しいリクエストはすぐ合流する。** プロンプトは decode の合間に読む: 短いプロンプトなら走っているストリームの停止は
  1 回約 12-20 ms。長いプロンプトは chunk（`STRATA_PAR_CHUNK`、プロンプト経路の chunk、1 つ約 160 ms）に分けて読み、chunk の中でも
  48 層の 8 層ごとに走っているストリームの decode を 1 ステップ挟む（`STRATA_PAR_YIELD`）。chunk の和は変わらず、8K のプロンプトを
  読む間の止まりは最長約 35 ms（前は 161 ms）、8K の最初のトークンも同じくらい早い（1.44 s。前は 1.50 s）。読み終えた会話の
  コピー（8K で約 75 ms）は他のストリームを止めずに裏で行う（`STRATA_PAR_SNAP_ASYNC`）。
- **会話の再利用は lane ごと。** 終わったリクエストの lane はその会話を持ち続け、新しいリクエストは「そのプロンプトが続きに
  なっている会話」を持つ空き lane に入る（エージェントが同じ文脈を送り直しても、新しい部分だけ読む）。
- **キャンセル**（クライアントの切断）はその 1 本だけを止める。
- `"parallel"` が 2 以上では使えないもの: drafter（`STRATA_INK_DRAFT`、`STRATA_INK_DSPARK`。1 つの会話を読むため）、画像、
  リクエストごとの無検閲化 / control vector の切り替え（エンジンの既定になる）、penalty、ディスク / RAM の会話スロット。
  プロンプト経路は expert pool の尾を借りる代わりに専用の 2 GB（`STRATA_PAR_PREFILL_MB`）を持つので、16K のプロンプトを
  単独で読むときは `"parallel": 1` より少し遅い。
- ほかのモデルは今はこの設定を無視する（1 本ずつ）。

## メモリと速度

エンジンは、密な重み（UD-Q2_K_XL で 6.2 GB）、キャッシュ、prompt 用のバッファを置いた残りの VRAM をエキスパートで
埋め、空き RAM から pinned の RAM 層を確保し、足りない分は必要なときに SSD から読む。使ったエキスパートを
`<pack>/expert_usage.txt` に記録し、次の起動ではそれを先に読み込む。

RTX 5090 + Ryzen AI Max+ 395（RAM 94 GB）での実測（UD-Q2_K_XL、コンテキスト 65536、全エキスパートが
VRAM 23 GB + RAM 70 GB に載った状態、iGPU レーンあり）:

| | |
| --- | --- |
| 生成（1,024 トークンの応答、文章 / コード、greedy） | 約 92〜95 tok/s（iGPU の代わりに CPU レーンなら約 55〜60） |
| 生成（swe-bench のコーディング問題 16 問 × 3、サンプリングあり、通し） | 約 100 tok/s（通しで 1 トークン平均 10.1 ms。1,024 トークンの応答では単発 1 トークン 11.1 ms、2 トークンの検証窓 14.6） |
| 生成（swe-bench のコーディング問題 60 問 × 5、サンプリングあり、通し） | 約 107 tok/s（1 問 100.5〜114.5。v1.3.27） |
| 生成（コンテキスト 125K トークン） | 約 65 tok/s |
| prompt（8K / 32K / 125K トークン、サーバー、2 回目） | 約 1,630〜1,760 / 約 3,100〜3,270 / 約 2,800 tok/s |
| 起動（重み + エキスパートの読み込み） | 約 1 分 |
| 65536 トークン分のアテンションキャッシュ | 0.44 GB |

（生成の行: Abyss v1.3.21、Q5_K の attention pack（上の「軽い dense pack」。配布 GGUF のままなら約 82〜84 と約 91〜94）。
prompt と 125K の行: v1.3.16〜1.3.18、配布 GGUF のまま。同じマシン・同じファイルで llama.cpp `bed0a8566` は `-ngl 99 -ncmoe 32` で生成 28〜30 tok/s、
prompt は ubatch 512 で 78 tok/s、ubatch 8192 で 660 tok/s。）各リリースの数値と測り方は
[CHANGELOG.md](../CHANGELOG.md) にある。

起動直後や、メモリがモデルより少ない環境では、エキスパートを SSD から読むので遅くなる。RAM 層のエキスパートは、
一部を CPU が計算し、残りを GPU が PCIe 経由で取り寄せる。その配分は起動時に測って決める。

**iGPU レーン。** AMD の APU（Ryzen AI Max+ 395 の Radeon 8060S）では、内蔵 GPU が CPU と同じメモリを約 240 GB/s
で読める（CPU のスレッドは約 75 GB/s）。ROCm が入っていて `libabyss_igpu.so` がビルドされていれば、エンジンは
起動時に RAM 層を内蔵 GPU に登録し、RAM 層のエキスパートを CPU の代わりに内蔵 GPU で計算する（コピーはしない）。
どちらで計算するかは起動時に両方を測って決め、ログに出す（`iGPU lane: an expert 0.046 ms vs 0.107 on the CPU's
16 threads`）。内蔵 GPU がはっきり速くない場合（1.5 倍、`STRATA_GLM_IGPU_MARGIN`）、正確な合算が有効な場合（Qwen3.8）、
ライブラリか HIP のデバイスがない場合は CPU で計算する。
prompt のまとめ処理でもこのレーンを使う。prompt のチャンクが必要とする RAM 層のエキスパートを、CPU のスレッドではなく
内蔵 GPU で計算する（8K の prompt で +40%）。レーンの呼び出しには期限がある（`STRATA_GLM_IGPU_TIMEOUT_MS`、20 ms）。
カーネルドライバが内蔵 GPU のキューを止めているとき（起動後の数分間、読み込みのあとにカーネルがメモリを回収する間に
起きる）は、遅れた呼び出しが終わるまで CPU のスレッドが引き継ぎ、起動時の計測はきれいに測れるまでやり直す。

**投機デコード（`STRATA_DS4_SPEC`、既定でオン）。** この GGUF にはドラフトモデルがないが、応答は prompt や応答の
前のほうに出てきた並び（コード、ファイルパス、ツール呼び出し）を繰り返すことが多い。エンジンは直近 3 トークンが
前に出てきた場所を探し、その続きを次の 1 トークンと一緒に 1 回で確かめ（検証の窓）、モデルの答えと一致する間だけ
採用する。出力は使わない場合と同じで（同じロジットに同じサンプラー。`STRATA_DETERMINISTIC=1` ではバイトまで同じ）、
速さだけが変わる。v1.3.145 から DeepSeek V4 0731 では 1 回の窓に最大 5 トークンを下書きする
（`STRATA_DS4_SPEC_ADAPT`）。繰り返しの各位置は当たり続けている間だけ使い、窓の幅は、この機械で幅ごとにかかる時間と
残るトークン数から決める（`STRATA_DS4_SPEC_COST`）。4〜6 列の窓は 2 列の窓とあまり変わらない時間で済む。1 回の時間の大半は
CPU レーンが計算する RAM 層の expert で、列どうしで同じ expert を多く使うため。RTX 5090 + RAM 123 GB、内蔵 GPU なし、
4K 前後の prompt + 512 トークン（貪欲）、3 回の中央値、v1.3.144（1 回の窓に 1 トークン）との比較:

| DeepSeek V4 0731 UD-Q2_K_XL | v1.3.144 | v1.3.145 |
| --- | :---: | :---: |
| Python のファイルの書き換え（型ヒント、prompt 約 3K トークン） | 73.3 tok/s | 88.6 tok/s（+20.9%） |
| swe-bench 風のバグ修正（t030） | 82.2 tok/s | 92.0 tok/s（+11.9%） |
| 英語の散文（4K の prompt） | 71.1 tok/s | 71.4 tok/s（+0.5%） |
| 日本語の散文（鉄道史） | 76.0 tok/s | 74.7 tok/s（−1.7%） |

DeepSeek V4.1 1017 も 1 回の窓に最大 5 トークンを下書きする（v1.3.146）。それまで V4.1 の窓は batched だけで、
全列の attention のあと、全列の CPU レーンの expert を GPU が待っていたため、広い窓は割に合わなかった（swe-bench の文で
+1.9%、書き換えで −0.3%）。いまは V4 と同じ pipelined：列を 2 つのグループに分けて別々のストリームに流し、片方のグループの
RAM 層の expert を CPU レーンが計算している間に、GPU がもう片方の attention を進める（6 列の窓 91.6 → 70.4 ms）。
同じ機械と測り方で、1 回の窓に 1 トークンとの比較:

| DeepSeek V4.1 1017 Compact JA256 | 1 回の窓に 1 トークン | 最大 5 トークン |
| --- | :---: | :---: |
| Python のファイルの書き換え（型ヒント、prompt 約 3K トークン） | 55.5 tok/s | 71.5 tok/s（+28.8%） |
| swe-bench 風のバグ修正（t030） | 64.4 tok/s | 77.1 tok/s（+19.8%） |
| 英語の散文（4K の prompt） | 61.6 tok/s | 62.2 tok/s（+1.0%） |
| 日本語の散文（鉄道史） | 66.8 tok/s | 66.3 tok/s（−0.7%） |

`STRATA_DS4_SPEC=0` で下書きをオフ、`STRATA_DS4_SPEC_ADAPT=0` で V4 / V4.1 も 1 回の窓に
1 トークンと採用率のゲートに戻る。GPU を分割しているときも使う（v1.3.26）。

## DSpark のドラフトモデル（DeepSeek V4 Flash）

v1.3.43 から、DeepSeek V4 Flash 0731 に DeepSeek の DSpark ブロックドラフトモデルを並べて動かせる。DeepSeek V4 の 3 層が
本体の 40〜42 層の出力を読み、次の 5 トークンをそれぞれの確信度つきでまとめて提案する。確信度 0.6 以上が続く先頭の下書きを、
そのトークンと一緒に 1 回の検証の窓に通し、本体のサンプリングと一致したものだけを残す。回答は本体が出すものとして変わらず、
変わるのは 1 トークンあたりの時間だけ。ファイルは公式のドラフトモデルを bleysg が GGUF にした
[`DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf`](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)
（6.94 GB、MIT。expert は Q2_K、Markov ヘッドは Q8_0）。

**オンにする。** `./setup-deepseek.sh --dspark` がダウンロードして SHA-256 を照合し、設定に書き足す:

```json
"env": {"STRATA_DS4_DSPARK": "/home/you/llm/apps/Abyss-data/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF/DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf"}
```

そのあとサーバーを再起動する（`../Abyss-tools/abyss.sh restart`）。`./setup-deepseek.sh --no-dspark`（またはこの行を消す）で
オフに戻る。起動時のエンジンのログの `dspark:` の行に、読んだファイルと使った VRAM が出る（`3 stages over target layers
40-42, 5 drafts a step; 6.48 GB on CUDA0 ...`）。使えるのは DeepSeek V4 Flash 0731 を GPU 1 枚で動かすときだけで、層の分割、
MTP ブロックとは併用できない（DeepSeek V4.1 は専用のドラフターのファイルを使う。[DeepSeek V4.1 Flash](#deepseek-v41-flash)）。そのときは `dspark: not loaded - <理由>` と出て、本体はドラフトモデルなしで動く。

**自動で読む（v1.3.145）。** `STRATA_DS4_DSPARK` がないとき、エンジンはモデルの置き場でファイルを探し
（`<models>/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF/DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf`。`--dspark` が置く場所）、
ドラフトモデルの 6.48 GB を引いても expert のプールにモデルの全 expert が入るときだけ読む。目安は 48 GB 以上のカード。
32 GB のカードでは読まない。そこではドラフトモデルが取る 6.5 GB の分だけ expert のプールが小さくなり、上の n-gram の窓の
ほうが速い（書き換えの作業でドラフトモデルあり 75.9、n-gram の窓 88.6 tok/s。swe-bench の文で 78.3 と 92.0）。ログには `dspark: the drafter beside the models (no STRATA_DS4_DSPARK) - on` または `- off` と
大きさが出る。`STRATA_DS4_DSPARK=0` でどこでも読まない。`STRATA_DS4_DSPARK=<ファイル>` なら VRAM に関係なく読む（これまで
どおり）。`STRATA_DETERMINISTIC=1` のときと、共有メモリ（DGX Spark）では自動では読まない（使うならファイルを指定する）。

**かかるもの。** ドラフトモデルは重みと expert をすべて VRAM に置く。6.48 GB で、その分 expert のプールが小さくなる。
v1.3.144 までは RAM 層をドラフトモデルなしのときの大きさのままにしたので、いちばん使われない約 6.5 GB の expert は SSD から
読んでいた。v1.3.145 からは、RAM 全体の上限（RAM の 73%、100 GiB 以上の PC では RAM − 24 GB）に余裕があれば RAM 層が
その expert も持つ（RTX 5090 + 123 GB の V4 0731 で pinned 70.0 → 76.5 GB、ディスク読みなし）。`STRATA_DS4_DSPARK_TIER_CAP=1`
で前の上限に戻る。RAM 94 GB の PC を守っているのはこの全体の上限で、そこで RAM 層が 73.9 GB を pinned にすると空きが 13 GB に
なり、8K の prompt に 11 秒ではなく 301 秒かかった。下書き 1 回は AI PRO R9700 で約 4.4 ms。

**同じトークン（v1.3.145）。** V4 0731 の 3 列以上の窓（DSpark のもの、`STRATA_DS4_SPEC` を 2 以上にした n-gram のもの）は、
`STRATA_DETERMINISTIC=1` でもトークンごとの答えから離れることがあった（4K の prompt で 16 トークン目）。窓の dense の GEMV が
トークン経路と違う順に足していたこと（今は `STRATA_DS4_VERIFY_EXACT`）、pipelined の窓で同じグループの 2 列が expert の
作業領域を 1 組で共有していたこと、捨てた列の compressor ring の行が、あとのブロックの終わりで読む行に重なりえたこと、の
3 つ。すべて直し、ドラフトモデルありの答えは `STRATA_DS4_DSPARK=0` と同じになった。

**速さ**（AI PRO R9700、OCuLink PCIe Gen4 x4。RAM 94 GB の Ryzen AI Max+ 395 の PC。UD-Q2_K_XL と Q5_K の attention pack、
サーバー経由）:

| | ドラフトモデルなし | DSpark |
| --- | :---: | :---: |
| swe-bench のコーディング 6 問（サンプリングあり、通し）、内蔵 GPU なし | 48.4 tok/s（v1.3.42） | 61.9 tok/s |
| 同じく内蔵 GPU レーンあり | 51.7 tok/s（v1.3.41） | 67.3 tok/s |
| 1,024 トークンの回答、コード / 散文（貪欲、内蔵 GPU なし） | 46.6〜48.6 / 47.2〜47.5 tok/s | 63.4〜66.4 / 50.5〜51.4 tok/s |
| prompt 8K / 16K | 850 / 978〜1,028 tok/s | 799 / 977〜1,000 tok/s |

swe-bench の問題ごとに +16〜32%。これらの問題では下書きの 68〜82% が受理され、1 回の確認で 2.7〜3.7 トークン進む（コードを
写す問題では 4.7）。散文の伸びがいちばん小さい（1 回 2 トークンほど）。検証の窓は 1 列あたり約 9 ms かかり、これは CPU
レーンが 1 トークン分の RAM 層 expert を計算する時間（8.4 ms。RAM の約 93 GB/s で約 93 個）とほぼ同じ。速さを決めているのは
ドラフトモデルではなく窓なので、速い GPU より速い CPU / RAM や内蔵 GPU レーンのほうが効く。NVIDIA では、借りた
DGX Spark（v1.3.114）で生成が 21.0 → 22.8 tok/s になり、下書きの 78% が受理された。

**1 回ごとの選び方。** 直前の 3 トークンが prompt か回答の前のほうに出ていれば、その続き（最大 5 トークン）を窓の下書きに
して、ドラフトモデルは動かさない。ただし、その繰り返しが当たり続けている間だけ: 毎回、繰り返しの提案を実際に出たトークンと
照らし、位置ごとの当たり率を追う。先頭の位置が 0.75 を下回るとドラフトモデルに切り替え、そうでなければ位置ごとの当たり率の
積が 0.5 以上の長さまで繰り返しを使う。確信度の高い下書きがなければ、そのトークンだけを処理する。回答ごとにエンジンの
ログに出る `ds4 dspark:` の行に、窓の数、下書きと受理の数、1 回で進んだトークン数が出る。

**設定項目**（設定ファイルの `env`）:

| 設定 | 既定 | |
| --- | --- | --- |
| `STRATA_DS4_DSPARK` | 自動 | `<ドラフトモデルの GGUF>`: DeepSeek の DSpark ドラフトモデルを読み、それで生成する。`0`: 読まない。指定なし（v1.3.145）: モデルの置き場のファイルを、読んだあとも expert のプールに全 expert が入るときだけ読む（48 GB 前後以上のカード。32 GB のカード + RAM 層では n-gram の窓のほうが速いので読まない）。`STRATA_DETERMINISTIC` と共有メモリでは読まない |
| `STRATA_DS4_DSPARK_TIER_CAP` | オフ | `1`: ドラフトモデルがあるとき、RAM 層をドラフトモデルなしの大きさのままにする（v1.3.43〜v1.3.144 の動き。押し出された約 6.5 GB の expert は SSD から読む）。オフ: RAM 全体の上限で RAM 層の大きさを決める |
| `STRATA_DS4_DSPARK_THR` | 0.6 | 下書きに必要な確信度。下回る最初の下書きで窓を打ち切る（swe-bench: 0.7 で 61.3、0.8 で 59.8 tok/s、0.6 は 61.9。ds4 は 0.7） |
| `STRATA_DS4_DSPARK_K` | 5 | 1 回の窓の下書きの上限（1〜5） |
| `STRATA_SPEC_SAMPLING` | オフ | （qwen35 / qwen35moe の NextN ドラフター：Qwen3.8-27B、9B の各モデル、Ornith）サンプリングの要求で投機的サンプリングを使う。ドラフターが下書きをサンプリングで引き、窓は各下書きを確率 min(1, p/q) で残し、外れたら max(0, p − q) から引き直す。出るトークンの分布は下書きなしと完全に同じで、残る下書きが増える。`1`: seed を指定しない要求、`2`: seed つきの要求（と CLI の `--seed`）も。この場合、同じ seed の答えは前と変わる。greedy には影響しない。RTX 5090、Qwen3.8-27B UD-Q4_K_XL、9 seed × 512 トークン：温度 1.0 で +1.5 / +2.5 / +1.8%（コード / 英語 / 日本語）、0.6 で −0.5%。NextN のドラフターでは効果が小さい。`STRATA_SPEC_SAMPLING_DTEMP=<f>`: ドラフターが温度の f 倍で引く（0.5 / 0.7 は良くならなかった） |
| `STRATA_Q35_COPY_K` | 7 | （NextN の下書き役がある dense の qwen35: Qwen3.8-27B、Qwen3.5-9B、MiMo-9B）文の繰り返しから写す下書きの最大数。写す元は、直近 4 回の 3 トークン一致のうち最も長いもの、または直前 8 トークンの表。本数は実測の当たり率と窓の時間から決め、NextN より得なときだけ使います。`0` で v1.3.170 より前（最大 5）。RTX 5090、Qwen3.8-27B L-Q4_0: edit +22.7%、swe-bench +52.2%、文章 +3〜6%。出力は下書きなしと同じ |
| `STRATA_Q35_DFLASH` | （収まるときはセットアップが書く） | Qwen3.8-27B: z-lab の DFlash 2 下書き役の GGUF（`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`）。ステップごとに実測の「トークン / ms」で NextN ブロックと選び分ける。モデルの FFN が一緒に VRAM に収まらないときは起動時に外す。RTX 5090 L-Q4_0: code +24.6%、think +20.3%、日本語は同じ（[Qwen3.8-27B](#qwen38-27bdense)）。未設定: NextN ブロックだけ |
| `STRATA_Q35_DFLASH_KEEP` | 0 | `1`: FFN が一緒に VRAM に収まらなくても DFlash 2 を残す（FFN の行が RAM に回る。16 GB のカードでは遅くなる） |
| `STRATA_Q35_DFLASH_PICK` | 0 | `1`: 毎ステップ DFlash 2、`2`: 毎ステップ NextN（比較用）。0: 実測で選ぶ |
| `STRATA_Q35_COPY_KEY8` | オン | （`STRATA_Q35_COPY_K` と一緒に）写す元を直前 8 トークンの表でも探す。`0` で 3 トークン一致だけ |
| `STRATA_Q35_VERIFY_ROWS` | オン | （dense の qwen35 の検証窓）列ごとの正規化と SwiGLU を 1 回の起動でまとめて計算（値は同じ。Qwen3.8-27B の 8 列の窓で 28.1 → 24.9 ms）。`0` で列ごとに起動 |
| `STRATA_DS4_DSPARK_CHAIN` | オフ | `<p>`: 1 つずつのしきい値の代わりに、確信度の積が p 以上の間の下書きを使う |
| `STRATA_DS4_DSPARK_NGRAM` | オン | 文中の繰り返しを先に使う（上の説明）。`0`: 毎回ドラフトモデル |
| `STRATA_DS4_DSPARK_NGRAM_P` / `_NGRAM_CHAIN` | 0.75 / 0.5 | 繰り返しを使う先頭位置の当たり率 / その長さを決める位置ごとの当たり率の積 |
| `STRATA_DS4_DSPARK_EXPERTS` | 256（全部） | ドラフトモデルの 1 段あたりの VRAM の expert 枠（8〜256。256 で 1 段 2.1 GB）。枠のない expert はその行から外し、下書きのあとによく使うものを入れ替える。128 / 64 で 3〜5 GB 返せるが、本体の VRAM ヒットは 2 ポイントしか上がらず、下書きは 30〜50% 遅くなる（24 GB のカード向けの選択肢。そこでは未計測） |
| `STRATA_DS4_DSPARK_UPLOADS` | 4 | 下書きのあとに 1 段あたり枠へコピーする expert の数（0〜8） |
| `STRATA_DS4_DSPARK_PROF` | オフ | `1`: 終了時の報告に、ドラフトモデルの GPU 時間を段階ごとに出す |
| `STRATA_DS4_DSPARK_LOG` | オフ | `<n>`: n 回の窓ごとに、平均の時間と幅を出す（stderr） |

## 実行時の無検閲化（`--uncensor`）

v1.3.71 から、動かしながらモデルの拒否の方向を取り除けます。リクエストごとに入り切りでき、ファイル（16〜82 KB）はモデル自身から
`--abl-build` で作ります。有効にする方法、ファイルの作り方と確かめ方、実測、以前からある方法は、すべて
[UNCENSOR.ja.md](UNCENSOR.ja.md) にまとめています。

## 設定項目

（設定ファイルそのもの、つまり場所とすべての項目は [SETTINGS.ja.md](SETTINGS.ja.md) にあります。）

エンジンへの環境変数。設定ファイルの `"env"` に書く（Maya の全項目は [docs/maya/README.md](maya/README.md#tuning)。
ここで特に関係するもの）:

| 設定 | 既定値 | |
| --- | --- | --- |
| `STRATA_GLM_WARM` | オン | `0`: 起動時に読み込まない（使われたエキスパートから載っていく。最初の応答が遅い） |
| `STRATA_GLM_RAM_GB` | 空き RAM から | pinned の RAM 層の大きさを GB で固定。大きくすると SSD の待ちは減るが、そのエキスパートは CPU のレーンが計算する。DDR4 のデスクトップの V4.1 で 80 → 88 GB にすると SSD の待ち 11.1 → 8.7 ms、レーン 9.4 → 10.6 ms、生成 +3%（docs/BENCH） |
| `STRATA_GLM_PIN_CAP_GB` | なし | テスト用。RAM 層とその前後の pinned のバッファの合計がこの GB を超えると確保に失敗させ、RAM より先に pin できる量が尽きる GPU ドライバー（Windows の AMD、#49）をまねる。そのときの起動の動きを確かめられる: 小さいバッファは RAM 層より先に pin し、RAM 層は残りに合わせて縮み、層の後に 512 MB の予備が入らないときは止まらずに層のスロットを返す（ログに `gave back ... GB of RAM-tier slots`） |
| `STRATA_GLM_EXPERT_HELPER` / `_GIB` | オフ / 13 GiB | DeepSeek V4.1、CUDA のみ: 2 枚目の GPU の番号（例 `1`）と、そこに常駐させるデコード用エキスパートの置き場の大きさ。`0` でオフ。プロンプトは 1 枚目のまま。GPU が無い、置き場が入らない、ヘルパーでエラーが出たときは CPU で計算します。分割しない native pack と CPU レーンが必要で、レイヤーグラフはオフのとき。DSpark の検証窓でも使います（v1.3.94）。[設定と制限（英語）](EXPERT_HELPER.md) |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6（ユニファイドメモリでは 16） | RAM 層を決めるとき、システムのために空けておく RAM。AMD のセットアップは config に 16 を書く |
| `STRATA_PAGEFILE_GROWTH` | on | Windows: RAM 層（とほかの空き RAM の判定）で、ページファイルがこれから広がれる分もコミットの空きとして数える。システム管理のページファイルは RAM の 3 倍か 4 GB の大きいほう（ドライブの 1/8 まで）、初期サイズ < 最大サイズで設定したものは最大サイズまで。どちらもドライブの空きから 1 GB を残した量を超えない。v1.3.174 より前は今のサイズだけを数えていたため、初期 8 GB / 最大 32 GB のページファイルでは RAM 層が空き RAM より数 GB 小さくなっていた。`0` で以前どおり今のサイズだけを数える。サイズ固定（初期 = 最大）のページファイルには影響しない |
| `STRATA_GLM_RAM_FRAC` | 0.73 | Linux・単体 GPU で `STRATA_GLM_RAM_GB` がないとき、RAM 層が pin する RAM の上限は「RAM 全体のこの割合」と「RAM 全体から `STRATA_GLM_RAM_KEEP_GB` を引いた量」の大きいほう（RAM が 100 GiB 以上の機械だけ。それより小さい機械は今までどおり割合だけ）。94 GB 機で 73% を超えて pin するとスラッシングした。これだけを設定すると割合だけで上限を決める（KEEP より前の動き）。0.05〜0.95 に収める（上限を外す値はログに一行出して丸める） |
| `STRATA_GLM_RAM_KEEP_GB` | 24 | Linux・単体 GPU で `STRATA_GLM_RAM_GB` がないとき、RAM 全体のうち RAM 層に使わずに残す量。ページキャッシュ（シャードや V4.1 の Engram 表）、サーバー、デスクトップのぶんで、RAM が大きくても増えないため、RAM が 100 GiB 以上の機械では 73% より多く pin する。RAM 123.5 GiB で上限 90.2 → 99.5 GB（RTX 5090 の DeepSeek V4.1 UD-IQ2_XXS は生成 30.2 → 34.5 tok/s）、251 GiB で 227 GB。64 / 96 GB の機械（62.5 / 94 GiB）は今までどおり 73%（45.6 / 68.6 GB）で、この変数を設定したときだけ KEEP を使う。エンジンの起動後にほかのアプリが 8 GB より多く増える使い方（重いデスクトップ、タブの多いブラウザなど）では `32` を設定する。空き RAM から `STRATA_GLM_RAM_HEADROOM_GB` を引いた量も超えない。どちらで決まったかはログの `RAM tier ... pinned` の行に出る |
| `STRATA_GLM_STREAM_FETCH` | オン | SSD から読んだエキスパートを、読めた塊から順に VRAM へコピーする。PCIe のコピーが SSD の読みのすぐ後ろを走る（v1.3.31。DeepSeek V4.1 で 1 トークン −9%）。`0` で読み終わってからまとめてコピー |
| `STRATA_GLM_RAM_THP` | オン（Linux） | RAM 層を透過的ヒュージページに置く（空きの 2 MB ブロックの範囲で。APU のレーンが再検証するページが 512 分の 1 になり、数秒の停止が消える）。`0` でドライバの pinned ページ |
| `STRATA_GLM_RESERVE_MB` | 1536。12 GB 以下の CUDA カードと、CUDA の Qwen3.8 は 700 | エキスパート用プールの横に空けておく VRAM。CUDA の Qwen3.8 は v1.3.132 からどのカードでも 700 です（DSpark がなく、プロンプトはプールから借りるため）。RTX 5090 でエキスパートの枠が 432 増え、4K + 512 の生成はドラフタなし 113.9 -> 121.1 tok/s、MTP 167.7 -> 175.7。CUDA も v1.3.82 から 12 GB を超えるカードでは 1536 です。RTX 5090 で DeepSeek V4.1 を `--max-context 65536` で動かすと、700 では attention と DSpark ドラフタのバッファで VRAM が足りなくなっていました。HIP は v1.3.68 から多めに空けます。最初のプロンプトのときに hipBLASLt の GEMM がカーネルを読み込み（約 0.7 GB）、0.7 GB の空きでは R9700 で `--max-context 131072` のときに読み込めず、プロンプトの GEMM がすべて失敗して 8K のプロンプトに 1,300 秒かかっていました。プールが小さくなる分、生成は約 3% 遅くなります。エンジンのフラグ `--vram-reserve-mib N`（設定の `args` に書く）は、どのモデルでも同じ空きを決めます。GLM-5.3 / Qwen3.8 / DeepSeek V4.1 のプールにも、古い qwen4exp の経路の `--expert-cache auto` にも効きます。v1.3.129 までは、このプールではフラグが無視されていました（#39）。両方を指定したときはフラグが優先です（この環境変数はセットアップがマシンごとの既定として書くもので、フラグは設定そのものの指定だから。ログに `NOTE: --vram-reserve-mib N overrides STRATA_GLM_RESERVE_MB=M` と出ます） |
| `STRATA_GLM_POOL_GB` | なし | エキスパート用プールの大きさの上限（GB）。A/B や、毎回同じスロット数にしたいとき（テストでは RTX 5090 で DeepSeek V4.1 JA256 を 21.6 = 層あたり 67 スロットに固定）。指定したときは `STRATA_GLM_POOL_PIN` は何も保存しません |
| `STRATA_GLM_POOL_PIN` | 1（オン） | エキスパート用プールの大きさを、カード + pack + 設定ごとに pack の横の `pool_pin.txt` に保存します（MiB 単位。キーは GPU の UUID、VRAM の大きさ、層、プールの前に持つバイト数（文脈の長さと KV の型）、ビルド、VRAM に関わる設定）。これで、その日にほかのプロセスが VRAM をどれだけ持っていても、毎回同じプールになります。以前は起動ごとの空き VRAM で層あたりのスロット数が決まり、RTX 5090 でほかが約 0.12 GB 多く持つだけで DeepSeek V4.1 Compact JA256 が 68 → 67 スロットになり、5 トークン目から答えが変わっていました。保存した値は、今の空きより 256 MiB 上までなら使います（はみ出しは空けておく分から）。それより足りないとき（ほかのプロセスが VRAM を持っているとき）は、その起動だけ空きから決めて 1 行知らせ、保存した値はそのまま次の起動に使います。保存した値より 1 GiB 以上空いていたら測り直します。`0`: オフ（毎回空き VRAM から）、`reset`: このキーの値を測り直す（`pool_pin.txt` を消すと全部測り直し）。`STRATA_GLM_POOL_GB` を指定したときと、unified memory では使いません |
| `--kv q4_0` / `k8v4` / `int8`（エンジンのフラグ。設定の `args` に書く） | `fp16` | Ornith-1.5 122B / 397B（`qwen35moe`、v1.3.134 から）: attention 層の K/V キャッシュを小さくし、空いた VRAM をエキスパートの置き場に回します。`q4_0` はアダマール回転のあと 4 ビット（fp16 の 28%）、`k8v4` は K を 8 ビット・V を 4 ビット（41%）、`int8` は 8 ビット（53%）。`--max-context 131072` のとき Ornith-1.5-397B のキャッシュは fp16 で 4.00 GB、q4_0 / k8v4 / int8 で 1.12 / 1.62 / 2.12 GB。16 GB のカードでは fp16 は `not enough VRAM for the expert pool` で止まり、3 つとも起動します（#44）。Ornith-1.5-122B（NextN ドラフターあり）は 3.25 → 0.91 GB（q4_0）。fp16 との KL（122B 24G64G、8K / 32K の文脈のあと 511 トークン）: q4_0 0.019 / 0.007（1 位のトークンが同じ割合 93.9% / 98.8%）、k8v4 0.013 / 0.005、int8 0.008 / 0.003。日本語の 8K の文脈のあとでは q4_0 0.031（1 位が同じ 91.6%）、k8v4 0.021、int8 0.014 で、nll は fp16 との差が 0.005 以内です。速さ（RTX 5090、122B 24G64G、NextN ドラフターあり）: 32 GB では生成は 4K〜64K で fp16 と同じ、プロンプトは 32K〜64K で 5〜9% 遅くなります。16 GB に制限して文脈 128K のときは、q4_0 でエキスパートに使える VRAM が 4.71 → 7.05 GB になり、32K のプロンプトが 746 → 1,654 tok/s、生成が 26.3 → 34.9 tok/s。 入るなら `int8`、VRAM が一番きついときは `q4_0`。Qwen3.8 の古い経路には PR #21 からあり、GLM-5.3 は `int8`（latent キャッシュ）だけ、qwen4exp の fast path は fp16 のままです。dense の Qwen3.8-27B は 3 つとも使えます。256K（RTX 5090、UD-Q4_K_XL、25 万トークンのコードのプロンプト）では fp16 の KV が 17.00 GB で FFN のうち 5.20 GB が RAM に出て生成 25.2 tok/s、`int8` は 9.03 GB で FFN が全部 VRAM に入り生成 38.4 tok/s、プロンプトは 1,410 tok/s（fp16 は 1,530。v1.3.175 から int8 のプロンプトの attention も fp16 と同じ 1 パスのテンソルコアのカーネルで動きます。以前は 599 tok/s）。Laguna は v1.3.175 から `int8` だけ使えます（256K で KV 12.35 → 6.56 GB。`q4_0` / `k8v4` は 1 行出して fp16 で動きます）。fp16 の KV のせいで dense の FFN が RAM に出るとき、またはエキスパートの置き場が 1 GB 以上かつ 10% 以上減るときは、起動時に `hint:` の行を 1 行出し、KV の大きさと `--kv int8` で空く量を示します。既定は fp16 のままです |
| `--kv-elastic 0` / `1`（エンジンのフラグ。設定の `args` に書く。`STRATA_GLM_KV_ELASTIC`） | `1` | Elastic KV（次のリリースから）: qwen35 系（Laguna、Qwen3.8-Flash-Next、Ornith、密な Qwen3.8-27B）を CUDA の GPU 1 枚で、`--max-context` が `--kv-min` より大きいときに使います。KV キャッシュのアドレス空間はコンテキスト全体ぶん予約しますが、VRAM は会話がそこまで来たぶんだけ（8192 位置ずつ）割り当てます。まだ使っていないぶんは重みに貸します。MoE では層あたりのエキスパート枠が増え、27B では CPU で動くはずだった FFN 層を層ごと VRAM にコピーして動かします。KV がその場所を必要とすると、貸したぶんから先に返します（エキスパートは冷たいものの上か RAM 階層へ移り、FFN 層は CPU に戻ります）。新しい会話になるとまた貸します。変わるのは重みをどこから読むかだけです。Laguna は v1.3.175 と同じトークンを出します（`STRATA_DETERMINISTIC=1 STRATA_GLM_CPU_EXACT=1` のとき）。27B は CPU で計算していた FFN の行を GPU で計算するため、VRAM の多いカードで動かしたときと同じ出力になります。RTX 5090、`--max-context 262144`、1K / 64K / 128K / 250K トークンのコードのプロンプトで、デコード tok/s（固定 -> elastic）: Laguna-S-2.1 UD-IQ4_XS 51.8 -> 75.3、39.5 -> 50.0、34.8 -> 35.5、20.4 -> 20.7（起動時のエキスパート枠 66 -> 120/層）。Qwen3.8-27B UD-Q4_K_XL 38.3 -> 130.3、37.0 -> 78.2、31.5 -> 55.8、26.3 -> 26.2（KV が約 128K を超えるまで FFN 8.25 GB が VRAM に戻る。1K のプロンプトは 860 -> 1,567 tok/s）。Laguna の 128K-250K のプロンプトは 7-8% 遅くなります（プロンプトの始めに、KV が取り戻すぶんのエキスパートを移すため）。ログ: 起動時に `glm elastic: ... KV on demand`、KV が増減するたびに 1 行。`/v1/status` の `kv` にも出ます。`0`: 従来どおり起動時にキャッシュを全部確保します（v1.3.175 と同じ出力・同じ速度）。HIP、複数 GPU への分割、ユニファイドメモリでは固定のままです（その旨を 1 行出します） |
| `--kv-min N`（エンジンのフラグ。`STRATA_GLM_KV_MIN`） | 32768 | `--kv-elastic 1` のとき、常に割り当てておく位置数（4096 以上）。32K ぶんの KV は 1.5 GB（Laguna）/ 2.1 GB（27B）と小さく、ふつうのチャットやエージェントの最初の数ターンはこの中に収まるので、会話の途中で重みが動くことはまずありません。`--max-context` が N 以下なら固定で動きます |
| `STRATA_GLM_CPU_LANE` | 物理コアごとに 1 スレッド（NVIDIA の内蔵 GPU、DGX Spark では無効） | RAM 層のエキスパートを計算する CPU スレッド数。`0` で無効 |
| `STRATA_GLM_CPU_LANE_SPARE` | 物理 16 コア以上で 2、それ未満は 0 | 生成中に CPU レーンが空けておくコア数（プロンプトには効きません）。8 コア以上で有効。12 コアでは 2 つ空けると 2.5-3.5% 遅くなり、16 コアでは GLM-5.3-Flash / V4 / V4.1 が 2-3% 速くなります（v1.3.142） |
| `STRATA_GLM_CPU_ONE_BATCH` | オン | CPU レーンがエキスパートの gate/up と down の行を 1 つのバッチで出し、down の行は自分のエキスパートの h だけを待ちます（GLM +2.9%、V4.1 +4.4%、8 コアの GLM +5.5%、Laguna 16 GB +10.6%。v1.3.142）。`0`: 従来の 2 バッチ |
| `STRATA_GLM_BG_LINK_GBS` | 7.2 | バックグラウンドのコピーの大きさを調整した回線（Gen4 x4）の GB/s。起動時の測定で回線がこの 85% を下回るとき、コピーを比例して小さくします（半分の速度にした回線で GLM-5.3-Flash +9.8%、V4.1 +12%。v1.3.142）。`0`: 無効 |
| `STRATA_GLM_DRAIN_WAIT` | 0（`STRATA_DETERMINISTIC=1` のときは 1） | `1`: トークンの区切りごとに、まだ転送中の VRAM → RAM の降格コピーをすべて待ちます。置き場所がトークンだけで決まるようになります（遅い回線では、待たないと VRAM のヒット率が回ごとに変わります）。v1.3.143 |
| `STRATA_GLM_DRAIN_SERVE` | 1 | 直前の区切りで降格したばかり（VRAM → RAM のコピーがまだ転送中）のエキスパートをルートが外したとき、サービスのスレッドがそのコピーを待たずに、出ていく途中の VRAM の枠から読みます。同じバイトなので答えは同じです。待ちの後ろには、その token のルートが全部並んでいました。ドライバーがまだ始めていないコピー（Windows の WDDM がコピーをまとめるとき、下書き中）だと全部が止まりました。`0`: 以前どおりコピーを待ちます。ディスクの道（CPU でエキスパートを読む）はいつも待ちます |
| `STRATA_GLM_CPU_XQ_PAR` | オン | CPU レーンが、呼び出しの活性値の量子化を 1 本のスレッドでなく、プールで最大 8 つに分けて行います（同じバイト。v1.3.144）。`0`: 1 本のスレッド |
| `STRATA_GLM_CPU_HQ_PIECES` | オン | CPU レーンが、各エキスパートの h の量子化を最大 8 つに分け、その行の gate/up を終えたジョブが受け持ちます。down の行の待ちが減ります（同じバイト。v1.3.144）。`0`: エキスパートごとに 1 ジョブ |
| `STRATA_GLM_CPU_BAND_SUM` | オン | CPU レーンの重み付き和を 256 行ずつの帯に分け、その帯の down を終えたジョブが受け持ちます（同じバイト。v1.3.144）。`0`: バッチの後に 1 本のスレッド |
| `STRATA_GLM_PREFILL_DISK_DEFER` | オン | プロンプト、CPU レーンのエキスパートがある層: SSD から読むエキスパートのグループを、読み終わったものから GPU に出します。CPU はすぐ計算を始められます（16 GB 4K で CPU の開始待ち 371 → 115 ms。出力は同じ。v1.3.144）。`0`: 元の順 |
| `STRATA_GLM_PREFILL_FLAT_DECAY` | オン | プロンプトの後、RAM 層と VRAM の追い出し・読み込みに使うルートの回数を、どの層も同じだけ古くします（それまでの回数はチャンク分の半減、チャンクの回数は 1 トークンずつ decode したのと同じ重み）。`0` で前のやり方: 層ごとに回数を足すたびに全層を半分にするので、8K のプロンプトの後は GLM-5.3 の MoE 42 層のうち 36 層の回数が 0 になり、その後の約 200 トークンは RAM 層がその層のエキスパートを先に捨てていました。RTX 5090（Gen4 x4）+ RAM 層 45 GB（64 GB の機械）、iGPU オフ、8K プロンプトの後 512 トークン、各 3 回の中央値: GLM-5.3 Maya-S-v2 の decode 25.6 → 27.7 tok/s（SSD の読み 6.32 → 5.67 回 / トークン）、V4.1 JA256 39.1 → 40.4 tok/s（3.55 → 3.37）、V4 0731 UD-Q2_K_XL 43.0 → 42.2 tok/s（6 回の中央値、SSD の読みは同じで、ぶれの範囲）。モデルが VRAM + RAM に収まる（SSD を読まない）ときは速さは変わりません。変わるのはエキスパートの置き場所だけで、1 トークンずつの経路の logits は同じビットです（Ornith-35B 16 GB、3,071 トークンの採点）。MTP の下書きありの greedy では、検証ウィンドウの計算が僅差のところで別のトークンを選ぶことがあります |
| `STRATA_GLM_PREFILL_PLAN_COPY` | オフ | プロンプト: `1` で、MoE 層ごとの計画の表をコピーで GPU に送る前のやり方に戻します。既定では kernel が host-mapped のまま読むので、ホストから GPU へのコピー engine（1 つ）で prestage のコピーの後ろに並びません（RTX 5090、Gen4 x4 で GLM-5.3 の 16K: 1591 → 1779 tok/s。出力は同じ） |
| `STRATA_GLM_PRESTAGE_EARLY` | オン | プロンプト: 層のコピーが終わったら、CPU レーンの計算中に次の MoE 層の prestage を 2 個ずつ送ります（出力は同じ。v1.3.144）。`0`: CPU の出力を送った後 |
| `STRATA_GLM_PREFILL_XQ8K` | オン | プロンプト、CPU レーンのエキスパートがある層: CPU に渡す x を、GPU で CPU と同じ計算の Q8_K にしてから送ります（同じバイト。量は float の約 28%）。`0`: float で送り、CPU で量子化 |
| `STRATA_GLM_CPU_EXACT` | `1`（x86 の CPU で Qwen3.8）、`0`（ほかのモデル。RTX 5090 で GLM-5.3 の生成が 14%、V4.1 が 5% 遅くなったため） | CPU レーンのエキスパートを、GPU のカーネルと同じ計算（GPU と同じ 8 ビットの活性、同じ整数の和）で求めます。どちらで計算しても同じ値になります。`0` で従来の ggml-cpu の計算。ARM の CPU は従来の計算のまま |
| `STRATA_GLM_LANE_XC` | `1`（x86 の CUDA で Qwen3.8）、`0`（ほかのモデル、HIP、ARM） | `STRATA_GLM_CPU_EXACT` と合わせて、CPU レーンのエキスパートが GPU とビット単位で同じ値になり（GPU と同じ足し算の順と、SiLU の exp）、層のエキスパートの和を固定小数点で取ります（重みを掛けた各エキスパートの行を 2^-34 単位の 64 ビット整数にして足し、最後に 1 回だけ float に丸める。どの順に足しても同じビットで、float の足し算と同じかそれ以上に正確）。どのエキスパートを CPU が計算したかで和が変わりません。prompt の CPU の分担もこれに合わせます（`STRATA_GLM_PREFILL_XROWS`）。これで `STRATA_DETERMINISTIC` なしでも、温度 0 の答えが毎回同じになります（Qwen3.8 で 1 チャンク約 2,000 トークンまでの prompt。それより長いチャンクは、出力の窓が埋まるたびに、エキスパートを走らせた順に足します）。`0` で従来の動き（CPU の分を別に足して最後に加え、prompt の分担は時間で決める）。エキスパートの先送り（`STRATA_GLM_DEFER`）と EXL3 のエキスパートでは使いません。ほかのモデル（GLM、Ornith、DeepSeek V4）は `1` で使えます（まだ測っていません） |
| `STRATA_Q4X_VERIFY_EXACT` | `1`（CUDA）、`0`（HIP） | Qwen3.8 の下書き（MTP、DFlash 2）あり: 検証窓の dense の GEMV を列ごとにトークン経路と同じ和にして、温度 0 の答えが下書きなしと同じになります（v1.3.125）。HIP では 1 列ずつの GEMV になり、dense の重みと出力ヘッドを列の数だけ読むため、AI PRO R9700 + MTP で 87 -> 67 tok/s に落ちました（#42）。そのため HIP は重みを 1 回だけ読む GEMV のままです（v1.3.125 より前と同じ。下書きありの答えが、長い答えの後半で下書きなしと分かれることがあります）。`1` / `0` でどちらかに固定。`STRATA_DETERMINISTIC` は `1` にします |
| `STRATA_DS4_VERIFY_EXACT` | `1`（CUDA）、`0`（HIP） | DeepSeek V4 / V4.1 の検証窓: dense の GEMV を列ごとにトークン経路と同じ和にする。`STRATA_Q4X_VERIFY_EXACT` と同じ（v1.3.145。1 行 1 warp のままだと、V4 0731 の 3 列の窓が `STRATA_DETERMINISTIC` でも 16 トークン目でトークン経路から離れた。`STRATA_DETERMINISTIC` は 1 にする） |
| `STRATA_Q35_VERIFY_EXACT` | `1`（CUDA）、`0`（HIP） | Qwen3.8-27B（dense）の検証窓: GEMV を列ごとにトークン経路と同じ和にする。`STRATA_Q4X_VERIFY_EXACT` と同じ（型ごとに 1 回の起動のままだと、UD-IQ2_XXS の n-gram の窓が 4K のプロンプトの 135 トークン目で下書きなしの答えと分かれた）。`0`: その起動 |
| `STRATA_GLM_CPU_CAL_RETRY` | 3 | 起動時の CPU レーンの計測が乱れていそうなとき（CPU の 4 分の 1 以上でほかの仕事が動いている、または一番遅い回が中央値の 3 倍を超える）、1 秒おきに最大この回数まで測り直し、一番小さい値を使います。それでも CPU が混んでいればログに出します。`0`: 1 回だけ |
| `STRATA_GLM_TAIL_ON_TOP` | 1 | VRAM に全エキスパート・予備・プロンプト処理が借りる分がすべて入るとき、借りる分をエキスパートの置き場とは別に確保し、プロンプトでエキスパートを追い出さないようにします。v1.3.119 から共有メモリ（DGX Spark）でも、メモリに余裕があればそうします（Qwen3.8 ISTA IQ3_XXS：32K のプロンプトのあとのデコードで SSD からエキスパートを読み直さなくなった）。`0`：借りる分をエキスパートの置き場から削ります（v1.3.104 より前の動き。共有メモリでは v1.3.119 より前の動き） |
| `STRATA_GLM_X3_SPLIT` | オフ | EXL3 の pack で、各ルートの RAM 層エキスパートのうちこの割合を CPU スレッドが、残りを iGPU が同時に計算する（`auto`: 起動時の計測から）。Ryzen AI Max+ 395 では電力と熱を共有するため遅くなる（[EXL3](exl3/README.md#settings)） |
| `STRATA_GLM_EXL3_DECODE` | GPU による | EXL3 の pack で、`new`（tensor core のタイル、RTX 30 以降）か `legacy`（CUDA コア）をどの GPU でも選ぶ |
| `STRATA_GLM_CPU_PIN` | オフ | `<CPU の並び>`（例 `0-15`、`0,2,4-7`）: CPU レーンのワーカーをその CPU に固定し、スレッド数を並びの数までにする。GPU 1 枚でも分割でも、Linux でも Windows でも効く（v1.3.29）。`1` / `0` は従来どおり、分割の各部のプールを固定するかどうか |
| `STRATA_GLM_PCIE_SHARE` | 起動時に測定 | RAM 層のエキスパートのうち、CPU で計算せずに PCIe で GPU に送る割合（`0` = すべて CPU、`1` = すべて GPU） |
| `STRATA_GLM_PCIE_SHARE_LATE` | なし | `<分担率>[:<ステップ数>]`：各リクエストは上の分担率で始まり、`<ステップ数>` トークン分（既定 256）進んだらこの分担率に切り替えます。分担率が低いと PCIe で送るエキスパートが減り、GPU のエキスパートの置き場が埋まりにくくなります。そこで回答の最初だけ PCIe で埋めてから切り替えます（issue #14）。`<分担率>:settle[:<窓>[:<伸び>[:<低下>]]]` と書くと、切り替えはリクエストごとではなくセッション全体で 1 回になります。`<窓>` トークン分（既定 128）ごとに VRAM のヒット率を測り、前の窓からの伸びが `<伸び>` ポイント（既定 1）未満になったらこの分担率に移ります。その水準から `<低下>` ポイント（既定 5）下がったら（話題が変わったとき）、もとの分担率に戻ります。置き場が温まるのは何回かの回答にまたがるので、リクエストごとの切り替えではそこまで届きませんでした。普通の `--bench` が速くなったときだけ残してください |
| `STRATA_GLM_LANE_LEAN` | 4 | CPU レーンがエキスパート 1 個を PCIe のこの倍数速くこなすなら、置き場が落ち着いた時点（上の `:settle` の切り替えを自動で）で RAM 層のエキスパートを全部 CPU に回し、それまでと VRAM のヒット率が落ちたときは測った分担に戻します。`0`: 測った分担だけ。手で書いた分担や plan が優先（[VRAM の小さいカード](SMALL-VRAM.ja.md)） |
| `STRATA_GLM_POOL_FLOOR` | 4 | エキスパート置き場の最小（層あたりの枠数。予備 + 常駐 1）。これ未満ではエンジンは起動を拒否し、VRAM が何 MB 足りないかを言います。v1.3.82 より前は route 1 本分（8）が床でした。その間の置き場では、route の残りを scratch 枠と CPU レーンでこなします（[VRAM の小さいカード](SMALL-VRAM.ja.md)） |
| `STRATA_GLM_PREFILL_KEEP` | 4 | 本来の残し（route + 予備 + 2）では prompt の chunk を貸せない置き場で、prompt の間に各層が残す枠数。残りは prompt 経路に貸し、終わった後に埋め直します（[VRAM の小さいカード](SMALL-VRAM.ja.md)） |
| `STRATA_GLM_PREFILL_SMALL_BELOW` | 32768 | 本来の残しで貸せる prompt の chunk がこのトークン数より短く、小さい残し（`STRATA_GLM_PREFILL_KEEP`）でその 2 倍以上を貸せるときも、小さい残しにします（16 GB の Ornith-1.5-397B 16G64G: chunk 1,024 -> 12,288、4K の prompt が約 3 倍速）。v1.3.146 から 32768（以前は 4096）：16 GB のカードで 16K の prompt を 3 つではなく 1 つの chunk で読みます（GLM-5.3-Flash 587 -> 1244 tok/s、DeepSeek V4 0731 1008 -> 1607。4K の prompt、24 / 32 GB のカード、V4.1 JA256 は変わりません）。ただし、エキスパートのバイトの 90% 以上が VRAM と RAM 層にあるときだけです。それ未満（ディスクから読むエキスパートがある）では 4096 のままにします。prompt の間に追い出した常駐が別のエキスパートをディスクに押し出すためです（16 GB・RAM 層 60 GB の GLM-5.3-Flash、リンク半分の速さ：prompt の後の 512 トークンが 17.8 -> 15.3 tok/s）。そこでも 1 つの chunk にするなら `32768` を書きます。chunk で変わるのは prompt の足し算の順番だけで、後に続くトークンの KL は 3 つの chunk のときと同じですが、greedy の答えは v1.3.145 と変わることがあります。`4096` で以前の chunk に戻ります。`0` で、本来の残しで chunk を貸せないときだけ（v1.3.136 の動き） |
| `STRATA_GLM_PREFILL_DENSE_TIER` | オフ | DeepSeek V4 / V4.1、GLM-5.3、qwen35moe 系（Ornith）、GPU 1 枚（issue #53。速さは V4.1 で測定）: `1` = prompt の chunk より長い prompt では、VRAM にある各層の dense の重みも作業領域として借ります。そこに prompt 経路のバッファを置き、各層の重みはその層を使う直前に（モデルファイルの mapping から、小さな pinned バッファ経由で）運び直します。次の層の分は今の層の計算中に運びます。prompt が終わったら借りた分をすべて書き戻すので、decode は前と同じ重みを読みます。chunk が長くなると、エキスパートを流す回数が減ります（SSD と PCIe の量が減る）（RTX 5090 を 16 GB に絞り、RAM の置き場 44 GB、iGPU なしの DeepSeek V4.1 UD-IQ2_XXS: chunk 12,544 -> 32,768、32K の prompt 約 440 -> 約 670 tok/s。context 262,144 では chunk 6,400 -> 32,768、128K の prompt 413 -> 1,374、250K 445 -> 1,662 tok/s。decode は同じ）。prompt ごとに運ぶ手間（0.9〜3.7 秒）と書き戻し（約 1 秒、6.4 GB）がかかります。読み込み後にモデルファイルの mapping を手放す環境（Windows）では、行を RAM の写しから運びます（このモデルで 6.6 GB）。`tools/dense_overlay.py` で作った pack（層の行が 2 か所に分かれる）でも使えます。今の chunk に収まる prompt では何もしません。chunk がもともと 32,768 の 32 GB のカードでは変わりません。長い chunk で変わるのは、ほかの chunk の大きさと同じく prompt の足し算の順番だけです。`2`（試験用）: バッチの prompt はすべて、今の chunk のまま dense tier を通ります。`0` とトークンが同じになることを確かめるためのものです（`STRATA_DETERMINISTIC=1`）。起動時の `dense tier ...` の行と、prompt ごとの 1 行（chunk、層の読み込み、重みを戻す時間）で何をしたかが分かります |
| `STRATA_GLM_SEED_TAIL` / `STRATA_GLM_SEED_ROWS` / `STRATA_GLM_SEED_MARGIN` | DeepSeek V4 0731 / V4.1 1017 は 16、ほかのモデルは 0（オフ）（v1.3.147）/ 128 / 2 | prompt の最後の chunk で、その後に続くトークンのために VRAM のエキスパート置き場を入れ替えます（GPU 1 枚）。MoE の層ごとに、chunk が RAM 層から運んだエキスパートのうち、最後の `_ROWS` 行がよく使ったものを最大 `STRATA_GLM_SEED_TAIL` 個、その行があまり使わなかった常駐（`_MARGIN` 回以上少ないもの）の場所に置きます。エキスパートは prompt が置いている運び先の枠から GPU の中でコピーし、追い出す常駐はそのエキスパートがいた RAM 層の枠へ戻します（tier は重ならないまま）。重みを動かすだけなので答えは同じです（決定論モードの md5 は変わりません）。RTX 5090、Gen4 x4、iGPU なし、`=16`：DeepSeek V4 0731 UD-Q2_K_XL 32 GB 4K で prompt の後の最初の 64 トークンが 54.6 -> 62.0 tok/s、V4.1 Compact JA256 42.4 -> 50.6、GLM-5.3 Maya-S-v2 IQ2_XXS 41.8 -> 44.0。戻すコピーが prompt 自身の運び込みとリンクを分け合うので、prompt が約 1〜1.6% 遅くなります（V4 4K 845 -> 832 tok/s、最初のトークンまで +0.08 秒）。それでも最初の 64 トークンは約 0.07 秒早く終わり、512 トークンは +4〜10% 速くなります。16 GB では prompt +0.3〜0.9%、最初の 64 トークン +4〜9% です。`=0` でオフになります。GLM-5.3 は既定オフです。16K の prompt では 512 トークンで遅くなります（`=32` で 47.9 -> 43.6 tok/s。長い prompt の後の文章は末尾のエキスパートから離れるため）。HIP では測っていません |
| `STRATA_GLM_PREFILL` / `--prefill` | `auto` | prompt のまとめ処理の大きさ: `auto`、トークン数（上限は 32768。それより大きい値は 32768 にして `NOTE:` の行を出す）、`0`（1 トークンずつ） |
| `STRATA_WATCHDOG_S` | 180 | serve の見張り役：この秒数のあいだ進みのないリクエストはエンジンを終了させ、サーバーが起動し直す（`0` でオフ）。長いプロンプトの最中に大きなファイルをコピーすると（メモリの逼迫。ここではネットワーク共有へ 200 GB）、エンジンが 1 か所で数分待たされ、見張り役が止まったと判断することがある。値を上げる（`600`）か、コピーを別の時間にする |
| `STRATA_GLM_PREFILL_SUB` | auto（NVIDIA）、1024（HIP） | prompt の mixer と dense FFN のサブバッチ（16〜8192）。HIP の既定は v1.3.40 から 1024（以前は 256。AI PRO R9700 で DeepSeek V4 の 8K / 16K の prompt が +8% / +10%、GLM-5.3-Flash が +13% / +15%） |
| `STRATA_GLM_PROMPT_SKIP` | on | DeepSeek V4.1、1 台の GPU：長い prompt の最後の 2,668 行より前の行は、20 層目の attention で止める（20〜39 層はそれらの行を 20 層目の圧縮 KV からしか読まない。エキスパートの足し算の順を除いて同じ結果。v1.3.58、AI PRO R9700 で 32K の prompt が 64 → 101 tok/s）。`0`：オフ |
| `STRATA_GLM_V41_BATCH` | on | DeepSeek V4.1 の prompt の attention：射影（q_a、kv、圧縮器、q_b、出力の LoRA）をサブバッチまとめて行列計算し、リング・圧縮行・indexer・attention は 1 トークンずつのまま（v1.3.59。skip と合わせて AI PRO R9700 の 8K / 16K / 32K / 60K の prompt が 86 / 101 / 101 / 101 → 185 / 242 / 273 / 262 tok/s、KL は同じ設定を 2 回回したときの揺れと同じ）。`0`：1 トークンずつの attention |
| `STRATA_GLM_V41_ATTN_TOK` | off | `1`：V4.1 の prompt の attention のリング・圧縮行・indexer・attention を 1 トークンずつに戻す（v1.3.59 の処理。比較用）。off：サブバッチまとめて計算（v1.3.60。下の Engram の行列計算と合わせて、AI PRO R9700 + iGPU で UD-IQ2_XXS の 8K / 16K / 32K / 60K の prompt が 237 / 292 / 314 / 277 → 363 / 596 / 781 / 896 tok/s、Maya-XS は 192 / 334 / 327 / 288 → 964 / 1055 / 1093 / 1028。1 トークンずつの版との KL は 8K〜40K で 0.0038〜0.0040） |
| `STRATA_GLM_V41_ENG_GEMM` | on | V4.1 の prompt：Engram の engram_kv を 256 トークンごとに 1 回の FP16 行列計算に（v1.3.60。トークンごとの GEMV は約 160 MB の重みを毎トークン読み直していた。32K の prompt で 20 → 3.9 秒）。`0`：1 トークンずつ |
| `STRATA_GLM_V41_CAND_ROW` | off | `1`：V4.1 の prompt の候補ブロック（indexer で見える行が 16K を超える行）を 1 行ずつに戻す。off：行のまとまりごとに 1 回の呼び出し（v1.3.63）。HIP の gfx11 / gfx12 では V4.1 の 32 ヘッドの indexer も WMMA で計算する（`STRATA_DS4_IDX=1` で戻す。40K の prompt で GPU 時間 3.4 → 1.9 秒）。両方で AI PRO R9700 の Maya-XS の 32K / 60K の prompt が 1,093 / 1,028 → 1,119 / 1,214 tok/s（UD-IQ2_XXS の 60K は SSD が上限で変わらず） |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | 起動をまたいで残すエキスパートの使用記録。`0` で無効 |
| `STRATA_GLM_PRIOR` | サーバが設定 | パックに無いときに使うルーティングの統計（`expert_counts.txt` の形式、`tools/glm_expert_prior.py`）。起動時はこの順にエキスパートを置き、層ごとの VRAM の割り当てもこれに従う。Abyss が統計を同梱しているモデル（Qwen3.8-Flash-Next、量子化は問わない）では、サーバが `data/expert_counts/` を指す（v1.3.86。RTX 5090、ISTA IQ3_S、上の入れ替え規則と合わせて `--bench` 87.6 → 116 tok/s） |
| `STRATA_GLM_NO_REUSE` | オフ | `1`: 会話を再利用しない |
| `STRATA_SNAP_AT` | オン | サーバー: 会話の状態を、プロンプトの最後のトークンの手前ではなく、最後のメッセージの終わり（GEN キー `snap_at=`）で取っておく。次のターンが前の答えを書かれたとおりに送り返さなくても、そこから続きを読む。`max_tokens` で考えている途中に切れて答えが空のターンは描画されないため、これまでは次のターンで会話を全部読み直していた（#59）。DeepSeek V4 / V4.1 は 1 トークン手前（テンプレートが続けて来たユーザーのメッセージをつなぐため）。`0`: 最後のトークンの手前（従来どおり） |
| `STRATA_GLM_SLOTS` | 4（GLM-5.3）、8（Qwen3.8 / Ornith / Qwen3.8-27B / Ling-3.2-air） | 会話スロット。エンジンが持つ会話は 1 つで、それを続けないリクエストが来ると、その会話をいったん退避し、続きにあたるスロット（トークン列がプロンプトの先頭と完全に一致する、いちばん長いもの）を戻す。交互に動くチャットや、Claude Code のメインスレッドとサブエージェントが、互いのプロンプトを読み直さなくなる。GLM-5.3 はファイル。Qwen3.8 / Ornith / Qwen3.8-27B はまずホストの RAM（`STRATA_GLM_SLOT_RAM_GB`）、あふれたらファイル。スロットには delta-net の状態と全アテンション層のキャッシュ行が入る（Qwen3.8 UD-IQ3_XXS、3.6 万トークンで 1.05 GB）。RTX 5090、Qwen3.8 UD-IQ3_XXS、3 万トークンのメインと 2 万トークンのサブエージェントが交互: 戻るたびのプロンプト 11.1 秒 → 0.2 秒（退避と復元に +0.1〜0.3 秒）。スロットから戻した状態は、その会話自身のスナップショットと同じ（同じトークンが続く）。Ling-3.2-air: KDA の状態と conv 行、MLA 層の潜在キャッシュ行で、79 MB + 1 トークン 8.2 KB（3.57 万トークンで 0.37 GB）。RTX 5090、bartowski Q4_K_M、1〜3.6 万トークンのスレッド 4 本が交互: 戻ったスレッドが答え始めるまで 7.1〜19.2 秒 → 2.1〜3.8 秒、どの回答もそのスレッド単独と同じ。会話は来たときの API キーのもので、別のキーのリクエストには戻さない。テキストのみ（画像を含む会話は退避しない）。DeepSeek V4 / V4.1 と、複数 GPU に分けたモデルでは無効。`0`: 無効（会話 1 つ、従来どおり） |
| `STRATA_GLM_SLOT_RAM_GB` | 自動 | Qwen3.8 / Ornith / Qwen3.8-27B のスロットが使う RAM（pinned のホストメモリ、RAM 層とは別）。自動 = 最初のスロットを退避したときの空きメモリの 4 分の 1、最大 8 GB。退避後にマシンの空きメモリが 8 GB を切るときは RAM に置かない。上限を超えると、いちばん長く使っていないスロットからファイルへ移す（戻すのに 1 GB あたり約 0.3 秒）。`0`: ファイルのみ（GLM-5.3 は常にこれ） |
| `STRATA_GLM_SLOT_GB` / `STRATA_GLM_SLOT_DIR` / `STRATA_GLM_SLOT_MIN` | 16 / `<pack>/slots` / 1024 | スロットのファイル。合計この GB まで（超えたら古いものから消す）、このフォルダ（`STRATA_GLM_SLOT_KEEP` でなければ起動時に空にする）、ドライブの空きが 8 GB を切るときは書かない。`_MIN` トークンより短い会話は退避しない |
| `STRATA_GLM_SLOT_KEEP` / `STRATA_GLM_SLOT_DAYS` | オフ / 7 | `1`: スロットをエンジンの終了後も残す。終了時に会話をファイルへ書き、次の起動で戻す（`_DAYS` 日使われなかったファイルと、別のモデルのファイルは消す） |
| `STRATA_GLM_THINK_RESERVE` | 1024 | （サーバー）思考が回答のために必ず残すトークン数（リクエストの `max_tokens` の半分まで） |
| `STRATA_GLM_END_GUARD` | 1 | （サーバー）ターンの終わりの見張り。思考の途中でターンが終わったら思考を閉じて続け、思考の直後に回答なしで終わったら次に確率の高いトークンで続ける（どちらも 1 回だけ。強く量子化したモデルの空の回答の対策）。`0` で切る |
| `STRATA_GLM_LOOP_LONG` | `1024` | 思考中の長いループの見張り。33〜N トークンの同じかたまりがトークン単位でそのまま 3 回続いたら（繰り返しが 256 トークン以上のときだけ）、思考の上限と同じやり方で思考を閉じて答えに移る（V4.1 が temperature 0 で同じ 143 トークンの列挙を 1.8 万トークン繰り返し、回答の余地で閉じるまで止まらなかった）。答えの部分は見ない。`0` で無効（短いパターンの見張りだけ、前と同じ） |
| `STRATA_GLM_THINK_BRIDGE` | （英語の短い文） | （サーバー）思考の上限やループの見張りで思考を閉じるとき、`</think>` の直前に入れる文。「ここで考えるのをやめ、質問の言語で答えを書く」という内容（考えの途中に `</think>` だけを入れると、Qwen3.8 が答えの中で英語で考え続けることがあった。issue #36）。別の文を指定すれば置き換え、`0` で前と同じく `</think>` だけ |
| `STRATA_ANSWER_ROOM` | 1 | （サーバー）回答の余白。思考オンのとき、残りが最大トークン数の 4 分の 1（`STRATA_GLM_THINK_RESERVE` の 1024 以上、半分以下）になってもまだ思考中なら、思考の上限と同じやり方（つなぎの文、続けて `</think>`）で思考を閉じる。上限付きのリクエストが空の `length` で終わらず回答で終わる（設定の `thinking_budget` があってもなくても。以前はあるときだけ）。`usage` に `reasoning_closed_by: "answer_room"` と出る。`0` で切る |
| `STRATA_SOFT_END` | 16 | （サーバー）思考中や書きかけのツール呼び出しの中に出たターン終わりのトークン（`<|im_end|>`、GLM の `<|observation|>`、DeepSeek の文の終わりなど）を文字として残し、回答を続ける。1 リクエストでこの回数まで（チャットテンプレートを引用したファイルをツールで書くときなど）。`0` で、こうしたトークンは常にターンを終える |
| `STRATA_LANG` | （なし） | （サーバー）すべてのリクエストの既定の言語プロファイル（`ja-en`、`zh-en`、`ko-en`、`en`。「回答の言語をそろえる」を参照）。ほかの文字体系のトークンを回答で禁止する。リクエストの `"lang"` が優先、`off` でなし |
| `STRATA_LANG_THINK` | 0 | （サーバー）`1` で言語プロファイルを思考にもかける（既定は回答だけ） |
| `STRATA_JA_SYSPROMPT` | 1 | （サーバー）DeepSeek V4 0731：システムメッセージがなく、最後のユーザーのメッセージに仮名があるリクエストに、システムメッセージ `日本語で答えてください。` を足す（「回答の言語をそろえる」を参照）。`0` で止める |
| `STRATA_KEEP_PAST_THINK` | 1 | （サーバー）2026-11 より前の Qwen3.5 系のチャットテンプレート（Ornith-1.5-122B、Qwen3.5-9B QSA）は、最後のユーザー発言より前のアシスタントの `<think>` を落とします。そのため 2 ターン目が 1 ターン目を使い回せず、会話を全部読み直していました。サーバーがそのテンプレートを新しい判定（`preserve_thinking` が false でなければ残す）に変えます。`0` = テンプレートのまま |
| `STRATA_OOMD_GUARD` | 1 | （サーバー、Linux）大きなモデルを読み込む途中で systemd-oomd がターミナルごと終わらせるのを防ぐ助け。空きメモリ（MemAvailable）がモデルのサイズの 0.8 倍より少なく、ユーザーのスライスにファイルキャッシュが 1 GiB 以上あるとき、エンジンを起動する前にモデルファイルのきれいなキャッシュページを捨てます（`[abyss] memory: released ...` を 1 行出します）。エンジンが動いている間は小さな `<ログ>.engine-run.json` を置き、次の起動でうまく終わらなかったエンジンのものが見つかると、そのことを伝えます（systemd-oomd のジャーナルに行があれば、それも出します）。`0` で両方とも止める |
| `STRATA_TOOL_NONE_NOTE` | 1 | （サーバー）ツール呼び出しを含む会話で `tool_choice: "none"` のとき、最後に「文章で答える」というユーザーのターンを足す（それでもモデルが書いた呼び出しは、呼び出しとしても記法の文字列としても送らない）。`0` で足さない |
| `STRATA_ENGINE_STALL_S` | 90 | （サーバー）この秒数、何も言わず CPU・ディスク・GPU も使わないエンジンは止めて、次のリクエストで起動し直す。`0` で無効 |
| `STRATA_ENGINE_START_S` | 1800 | （サーバー）起動から READY までの上限（SSD から読むモデルは数分かかる）。超えるとエンジンを止め、起動失敗として理由を 1 行出す。`0` で無制限（v1.3.78） |
| `STRATA_ENGINE_DRAIN_S` | 120 | （サーバー）途中で終わったリクエスト（クライアントが切った）は DONE まで読み捨てる。STOP のあとこの秒数黙ったエンジンは止める。`0` で無制限（v1.3.78） |
| `STRATA_ENGINE_WRAP_S` | 40 | （サーバー）Ctrl+C / SIGTERM のとき、進行中のリクエストは思考を閉じて回答するのをこの秒数まで待ち、そのあと打ち切る |
| `STRATA_ENGINE_QUIT_S` | 50 | （サーバー）QUIT のあとエンジンが終わるまで待つ秒数（`STRATA_GLM_SLOT_KEEP` では先に会話を書き出す）。超えると kill。llama-swap の unloadTimeout はこれより長く |
| `STRATA_VISION_START_S` / `STRATA_VISION_ENCODE_S` | 600 / 120 | （サーバー）画像エンコーダーの起動と 1 枚ごとの応答の上限。超えると止めてリクエストはエラー、次の画像で起動し直す（v1.3.78） |
| `STRATA_HTTP_BACKLOG` | 256 | （サーバー）listen のバックログ（accept 待ちのリクエスト数） |
| `STRATA_MAX_BODY_MIB` | 256 | （サーバー）リクエスト本文の上限。超えると読まずに 413 |
| `STRATA_HTTP_STALL_S` | 120 | （サーバー）ソケットの 1 回の操作が待てる秒数。リクエストの読み込み（本文が届かなくなったら 408）と、答えの 1 回の書き込み。要求全体の上限ではないので、長い prompt や長いストリームは切れない。ストリームを読まなくなったクライアント（スリープした PC、切れた VPN）がエンジンを握り続けなくなる。その答えは切断と同じく取り消され、次のリクエストが動く。`0` で上限なし |
| `STRATA_FIT_FLOOR` | min(1024, 文脈 / 4) | （サーバー）`fit_max_tokens` が文脈に収まらない max_tokens を縮めるとき（Anthropic の API の既定）に残す、答えのトークン数の下限。これより少ないと 400 `prompt is too long: N tokens > M maximum`（Anthropic の形。OpenAI の API では `context_length_exceeded`）で断り、Claude Code はこれを受けて会話を compaction する。max_tokens 無しのリクエストは今まで通り文脈の残り全部。`0` で 1 トークンまで縮める（v1.3.172 より前の動き） |
| `STRATA_RAW_SPECIALS` | 0 | （サーバー）`1`：メッセージ中の特殊トークンの文字列を再び制御トークンにする（以前の encode） |
| `GPU_MAX_HW_QUEUES` | 2（HIP） | ROCm がデバイスごとに作る HW キューの数。シェルにも設定ファイルにもないとき、サーバーが 2 を渡す（v1.3.40）。AI PRO R9700 の DeepSeek V4 の 8K / 16K の prompt は、1 で 586〜650 / 676〜684 tok/s（R9700 向けの手順でアイドル電力の回避策として広まっている値。この PC ではどちらでもアイドル 12 W）、ROCm の既定の 4 で 653〜737 / 748〜756、2 で 866〜995 / 999〜1,011 |
| `STRATA_GLM_IGPU` | 自動 | iGPU レーン。`libabyss_igpu.so` と HIP のデバイスがあり、CPU よりはっきり速ければ使う（`STRATA_GLM_IGPU_MARGIN`）。`0` で使わない、`1` で遅くても使う。`STRATA_GLM_IGPU_LIB=<パス>` で別のライブラリ |
| `STRATA_GLM_IGPU_MARGIN` | 1.5 | 自動のとき、起動時の計測でエキスパート 1 個の時間が CPU のこの倍以上速いときだけ iGPU がレーンを受け持つ（計測には実際のトークンでの iGPU 呼び出しの費用が入らない。Ryzen AI Max+ 395 + RTX 5090 の Qwen3.8 16G64G は 0.022 ms 対 0.019〜0.025 ms のほぼ同着で、iGPU を選んだ回はどれも decode が遅かった: MTP 141 → 132 tok/s、既定 94 → 89。GLM-5.3 / V4 / V4.1 は iGPU が 1.9〜2.7 倍速く、iGPU のまま）。正確な合算（`STRATA_GLM_LANE_XC`、Qwen3.8 の既定）が有効なときは CPU がレーンを持つ（iGPU の和は GPU と同じビットにならず、答えがまたレーンに左右されるため）。`1` = 単純に速い方（以前の選び方） |
| `STRATA_GLM_IGPU_TIMEOUT_MS` | 20 | これより遅れたレーンの呼び出しは CPU のスレッドが引き継ぐ（0: 待つ） |
| `STRATA_GLM_IGPU_POKE_MS` | 500 | ルーティングのあと、iGPU を空のカーネルで起こしておく時間（50 µs ごと。v1.3.40 で 20 から変更）。DeepSeek V4.1 はルーティングの間に SSD を 20 ms 以上待つことがあり、その間に寝た iGPU は次の呼び出しに 21〜27 ms かかっていた（AI PRO R9700 + 8060S の最も遅い系列: 遅延 123 回、1 token 80.4 ms → 500 で 3 回、65.6 ms） |
| `STRATA_GLM_IGPU_COLS` | オン | 検証窓（DSpark / 繰り返しのドラフト）の iGPU レーンのエキスパートを、窓の全列で 1 回のレーン呼び出しにまとめ、同じエキスパートを使う列が何本あっても 1 回だけ読む（v1.3.89。結果は同じビット。RTX 5090 + Radeon 8060S、DeepSeek V4.1 UD-IQ2_XXS + DSpark: レーン 60〜62 → 54〜57 us/エキスパート、decode −3〜5%/トークン、swe-bench 48.6 → 51.0 tok/s）。`ABYSS_IGPU_COLS_UNION=0`: 1 回の呼び出しの中で列ごとのグリッド（読みの共有なし）。`0`: 列ごとにレーンを呼ぶ |
| `STRATA_GLM_IGPU_EARLY` | オン | SSD からエキスパートを読むルートで、iGPU レーンを読みの後ではなく前に起動する（v1.3.36。DeepSeek V4.1 で −2%、出力は同じ）。`0`: 読みの後 |
| `STRATA_GLM_IGPU_XQ` | オン | トークンの入力を、GPU が量子化した q8_1 のまま request で受け取り、iGPU で量子化し直さない（v1.3.36、ビット同一）。`0`: iGPU で量子化 |
| `STRATA_GLM_PREFILL_IGPU` | オン | prompt の RAM 層エキスパートを iGPU で計算する。`0`: CPU のスレッド |
| `STRATA_GLM_PREFILL_CPU` | オン | prompt で CPU も計算する（PCIe で送る間に、RAM 層のエキスパートの一部をホストで計算）。`0`: すべて PCIe で送る |
| `STRATA_GLM_PREFILL_XROWS` | 32 | `STRATA_GLM_LANE_XC` のとき: チャンクの中でこの行数以下のエキスパートを、生成のカーネルと同じ計算で求めます。VRAM にあれば GPU で、RAM 層にあれば CPU で（時間に関係なく全部）。どちらでも同じビットです。CPU はそれより大きいエキスパートを取りません（大きいものは MMQ）。これで prompt の結果が、何が VRAM にあったかや、CPU の分担の学習によらなくなります（ディスクから読むエキスパートは MMQ のまま）。`0` で時間で決める分担（1-32） |
| `STRATA_GLM_PREFILL_PIN_MB` | なし | CPU の分の pinned バッファ 1 つの上限（MB）。収まらない、確保できない、または空き RAM が 2 GB を切るときは、それ以降の prompt はすべてのエキスパートを PCIe で送る（#31）。`0` は `STRATA_GLM_PREFILL_CPU=0` と同じ |
| `STRATA_GLM_IGPU_FUSED` | オン | prompt の iGPU エキスパートを fused な int8 tensor core カーネルで計算する（gfx1151、IQ2_XS / IQ3_XXS の 4096 / 2048 エキスパート。v1.3.18）。`0`: 1 行ずつのカーネル。`ABYSS_IGPU_FUSED_CHECK=<n>` で最初の n 回を 1 行カーネルと比べる（stderr） |
| `STRATA_GLM_MV_Q6K_SPLIT` | GPU ごと | `1`: 4,096 block までの Q6_K GEMV を 1 行 4 warp、`0`: 1 warp。未設定: エンジンが GPU ごとに選ぶ（gfx12 は 1 warp。R9700 で DeepSeek V4.1 の 1 トークンあたり 8.8 -> 7.5 ms。ほかは 4 warp）。`./abyss.sh --gpu-tune` が測る（v1.3.51） |
| `STRATA_GLM_MV_SPLIT` | auto | dense GEMV の 1 行を 4 warp で分担する: 未設定 = 256 block 未満と、Q6_K は 4,096 block まで（v1.3.18）。`0` 常に 1 warp、`2` / `4` 常にその分担。1 warp のときとビットは一致しない |
| `STRATA_GLM_MV_FUSE` | on（HIP） | 型の違う小さな GEMV の組（DeepSeek V4 の x の射影）を 1 回の起動で、1 行 4 wave で計算する（v1.3.40。AI PRO R9700 では小さな GEMV が大きさによらず 1 回 15〜29 µs かかる）。`0`: 型ごとに起動。型ごとのカーネルとはビットが一致しない |
| `STRATA_GLM_GRID_PAD` | on（HIP、gfx11 以外） | wave 数が GPU の wave スロット数の倍数ちょうどかその直前になる起動に、空のブロックを足す（v1.3.40）。gfx1201 ではそういう起動の次の起動が 8〜14 µs 遅れ（ROCm/TheRock#9300）、DeepSeek V4 の 4096 行の GEMV と expert の down が 1 token に約 275 回当たっていた（AI PRO R9700: 22.2 → 20.2 ms/token）。ビット一致。gfx11 では既定でオフ（issue #9、RX 7900 XT で遅れは出ず、差は誤差の範囲。v1.3.54）。`0` ですべてオフ、`1` ですべてオン |
| `STRATA_GLM_HC_CPU` | on（HIP）、off（NVIDIA） | deepseek4: MoE の層のうち CPU / iGPU レーンが計算した分を、専用の待ちカーネルではなく次の層の attention 側の mHC 読み出し（hc3）の中で足す（1 層あたり起動が 1 回減り、答えは hc3 の 32 ブロックで読む。v1.3.42）。ビット一致。`0` で待ちカーネル、NVIDIA でも `1` で有効（未測定） |
| `STRATA_GLM_HC3_V41` | on（HIP）、off（NVIDIA） | DeepSeek V4.1: 5120 幅の mHC 読み出しを hc2 ではなく hc3（40 ブロック、遅らせた collapse）で行う。変わるのは足し算の順だけ。RTX 5090、V4.1 JA256、p4k + 512 トークン、n-gram の下書きなし：生成の中央値 65.1 → 67.9 tok/s（+4%、各 3 回）。ただし 8 本の held-out 系列の KL が 8 本中 7 本で上がった（平均 0.3815 → 0.3839）ので、NVIDIA は hc2 のまま。`1` で NVIDIA でも hc3、`0` で HIP も hc2 |
| `STRATA_Q4X_PLE_GGUF` | pack のもの | Qwen3.8-Flash-Next: PLE 表（`per_layer_token_embd`）をこの GGUF から読む。IQ4_NL、Q5_1、Q8_0、BF16（Q5_1 / BF16 は v1.3.117 から）。RTX 5090、16G64G（M16b）、NVFP4 版との KL: IQ4_NL 0.0826、Q5_1 0.0732、Q8_0 0.0714、速さは同じ。表は 1 行ずつ読むので、大きい型で増えるのはディスクで、VRAM は増えない |
| `STRATA_Q4X_PLE_PREFETCH` | on | Qwen3.8-Flash-Next で MTP ドラフターを使うとき: 検証窓の PLE 表の行（1 トークン 16 回の SSD 読み）を、ドラフターがそのトークンを出した時点で行キャッシュに先読みする（窓が使う 1 下書き分前、v1.3.87）。RTX 5090、ISTA IQ3_S: 窓の行の待ち 0.8〜1.1 ms → 0.1〜0.25 ms、ドラフターあり +3%。`0` で窓の開始時に読む |
| `STRATA_Q4X_HC_FUSE` | on（NVIDIA）、off（HIP） | Qwen3.8-Flash-Next: 1 トークンの hyper-connection の混合（97 回）を 5 回ではなく 2 回の起動で行う。結果は同じビット（v1.3.86。`hc4_fused_parity` で確認）。RTX 5090、ISTA IQ3_S: 混合 2.98 → 2.21 ms/トークン、1 トークンずつの経路 +8%。v1.3.87 から MTP の検証窓も（1 窓 97 回の混合が 2 列で 2.6 → 1.8 ms、5 列で 5.5 → 2.6 ms。ドラフターあり +8%）。`0` で 5 回の起動、`2` でトークンは融合・窓は 5 回（比較用） |
| `STRATA_Q4X_GDN_FUSE` | on（NVIDIA）、off（HIP） | Qwen3.8-Flash-Next: delta-net 層の中間（q / k の L2 norm とゲート、再帰、ゲート付き出力 norm とその q8_1）を 4 回ではなく 1 回の起動で行う。1 トークンずつの経路と MTP の検証窓の両方（窓はゲートの後の 3 回を 1 回に）。結果は同じビット（`gdn_core_parity`）。RTX 5090: 36 層で単体 0.44 → 0.33 ms/トークン、decode は約 +1%（bench の揺れの範囲）。`0` で別々の起動 |
| `STRATA_Q35_QSA` | GGUF に QSA indexer があれば on | QSA indexer テンソル（`attention.indexer.*`、例: higa2515/Qwen3.8-27B-QSA-GGUF）を持つ dense の qwen35 GGUF: top_k + 3 セルを超えると、各 full-attention 層は indexer が選んだ上位 top_k / 4 個の 4 セルブロックと自分のブロックだけに attention する（top_k は `attention.indexer.top_k`。27B のファイルは 2048 = 512 ブロック。プロンプト・decode・MTP 検証窓すべて。KV は全部保持。層数・ヘッド数は GGUF から読む）。`attention.compress_ratios` が 0 の attention 層は dense のまま。`0` = 通常ファイルと同じ dense attention（この機能以前と同じ出力）、`2` = indexer のキーは作るが attention は dense のまま（テスト用）。`--kv` で KV を量子化したときと画像入りのリクエストでは dense。AMD（HIP）では v1.3.170 から有効（それまでは dense）。 |
| `STRATA_Q35_QSA_DENSE_LAYERS` | 未設定 | 上の QSA 経路のテスト用: ファイルに indexer があっても dense のままにする attention 層をカンマ区切りで書く（例: `3,31`。同じファイルのまま、その層の indexer は使わない）。出荷するファイルはメタデータで指定する: attention 層の `attention.compress_ratios` が 0 = その層は dense（Qwen3.5-9B QSA は 3 層をこの形で dense にしている。[Qwen3.5-9B QSA](#qwen35-9b-qsa)）。未設定: ファイルの ratios のまま。 |
| `STRATA_Q35_QSA_PICK` | 未設定 | 上の QSA 経路のテスト用（通常は使わない）: `recent` は indexer の選択の代わりに直近 512 ブロック（スライディング窓）、`first` は先頭のブロックを使う。 |
| `STRATA_Q4X_QSA_BATCH` | on（NVIDIA）、off（HIP） | Qwen3.8-Flash-Next のプロンプト: QSA 層の attention のうち、先頭 2,051 セルより後の行を 64 行まとめて 1 回の起動で行う（各行は自分の選んだセルを KV キャッシュから直接読む）。従来は 1 行ごとにセルのコピーと attention の起動。サブバッチの indexer キーも 1 行 1 回ではなく 2 回の起動で追加する。結果は同じビット（`qsa_rows_parity` で両方を確認）。RTX 5090、ISTA IQ3_S: QSA 段 2,000 → 346 ms（bench のプロンプト 12,353 トークン）、8K プロンプト 1,555 → 2,360 tok/s（+52%）、32K 1,710 → 3,015 tok/s（+76%）、2K +5%。`0` で 1 行ずつ |
| `STRATA_Q4X_QSA_DECODE` | on（NVIDIA）、off（HIP） | Qwen3.8-Flash-Next の decode: QSA 層の attention（1 トークンずつの経路と MTP の検証窓）で、セルのコピー + attn_decode をやめ、KV キャッシュから直接読み、1 ブロックで数ヘッドを処理する（窓は列をまとめて 1 回の起動、選択も 1 回、indexer キーは 2 回の起動）。結果は同じビット（`qsa_rows_parity`）。RTX 5090、1 層 2,051 セル: 1 トークン 22 → 18 us、4 列の窓 88 → 24 us、8 列 177 → 34 us。--bench の decode（ドラフターあり）126.9 / 128.9 → 131.0 / 131.3 tok/s（約 +2.6%、bench の揺れの範囲内）。`STRATA_Q35_ROWS_HPW`（1 ブロックのヘッド数: 1 / 2 / 3 / 4 / 6。既定は 1 行で 2、5 行まで 4、それ以上 6）は調整用。`0` で従来の起動 |
| `STRATA_GLM_WAIT_FETCH` | on（HIP）、off（NVIDIA） | MoE の層のディスク読み込みの待ちと、RAM 層の expert のコピーを 2 回ではなく 1 回の起動で行う（v1.3.42。処理は同じ）。`0` で 2 回の起動 |
| `STRATA_HIP_EXPERTS_LEGACY` | off | `1`: decode の expert を RDNA 版ではなく元のカーネルで計算する（gfx1100、v1.3.40 から gfx1201 も: AI PRO R9700 で gate/up と down が約 2 割速い。gfx1151 は IQ3_XXS の down 以外）。どちらでもビット一致 |
| `STRATA_HIP_OWN_GEMM` | on | HIP ビルド: プロンプト経路の行列積（FP16 / BF16 / F32 の重み x トークン、F32 で足す。全モデルの dense の射影、GLM の MLA のヘッドごとの積、DeepSeek V4 / V4.1 の indexer と Engram、Hy4、MMQ を使わないときの dense FFN）を、hipBLAS / rocBLAS / hipBLASLt ではなくエンジン自前のカーネルで計算する（issue #49: TheRock ROCm 7.14.1 で `hipblasGemmEx` -> `rocblas_gemm_ex` -> `hipblasLtMatmul` の中でアクセス違反）。RDNA3 / RDNA3.5（gfx11）と RDNA4（gfx12、`STRATA_HIP_GEMM_WMMA12` を参照）は WMMA、そのほかのカードは素の FMA カーネル。変わるのは足す順番だけ（ライブラリ経路との KL はぶれの範囲）。最初のプロンプトで `hip gemm: HIP0 (gfx1151) on the own GEMM ...` と出る。`0`: これまでどおり hipBLAS（`STRATA_HIPBLASLT_TUNING` もまた読む）。CUDA は変わらない |
| `STRATA_HIP_GEMM_WMMA12` | on | RDNA4（gfx1200 / gfx1201: RX 9060 / 9070、AI PRO R9700）で、自前 GEMM の 16 bit の積を WMMA のカーネルで計算する（#49、RX 9070 XT: `own_gemm_test --lib` 40 / 40、Ornith-1.5-122B の 5K / 19K / 77K の prompt が素のカーネルの 901 / 868 / 596 から 1,612 / 1,471 / 830 tok/s）。`0`: 素の FMA カーネル |
| `STRATA_HIP_EXPERTS` | auto | HIP ビルド: プロンプトで routed expert（MoE モデル）の積に使うカーネル。`auto`: まとめて読む最初のプロンプト（32 トークン以上）で、1 層の expert（VRAM にあるもの、1 個あたり 32〜128 行。まず 4 個で測り、MMQ が 5 ms 未満のときだけ 32 個で測り直すので、遅い MMQ を大きいサイズでは走らせない）を MMQ、自前の int8 GEMM（WMMA、RDNA3。RDNA4 は実機で動かしたことがないので測らず、選ばない）、FP16 経路（expert ごとに FP16 に 1 回展開して自前の FP16 GEMM）で 1 回ずつ測り、ほかの経路が 1.5 倍より速く、出力が MMQ と合う（相対 L2 が自前の GEMM は 0.02 未満、FP16 経路は 0.1 未満）ときだけ MMQ から切り替える。選んだカーネルが扱えない層は、信頼できるもう一方に回す（`own` のときは FP16 経路、`fp16` のときは測って合った場合だけ自前の GEMM）。どちらも扱えなければ MMQ。gate と up の型が違う層、EXL3、NVFP4 の expert の層は測らない（次の MoE 層で測る）。ログに `glm prefill: HIP0 expert kernels (layer ..): MMQ .. ms, own GEMM .. ms, FP16 .. ms -> ...` と出る。issue #49: Windows の RX 9060 XT（TheRock ROCm 7.14.1）で MMQ が FP16 の約 390 倍遅く（dense の層）、Qwen3.8-Flash-Next 16G64G のプロンプトが 9 tok/s だった。`mmq` / `own` / `fp16` は測らずにその経路に決める（どのカーネルが遅いかを切り分けるため）。`STRATA_PROMPT_MMQ` か `STRATA_PROMPT_OWN_TYPES` を設定したときも測らない。`STRATA_HIP_EXPERTS_CHECK_SCALE`（試験用）は測った MMQ の時間に掛ける倍率。CUDA は変わらない |
| `STRATA_HIP_GEMM_PORTABLE` | off | `1`: どのカードでも（gfx11 でも）自前 GEMM の素のカーネルを使う。gfx12 が既定で通る経路を比べるためのもの |
| `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN` | 4 / 16（Qwen3.8: 2 / 8） | RAM 層のエキスパートの使用回数が「VRAM 側の ratio 倍 + margin」を超えたら入れ替える（2 / 4 は以前の、入れ替えの多い規則）。Qwen3.8-Flash-Next は早めに入れ替え、1 回のトークン境界で最大 16 個動かす（`STRATA_GLM_PROMOTE`、他は 8）。エキスパートが約 2 MB で、1 個のコピーが約 0.3 ms のため（v1.3.86。RTX 5090、ISTA IQ3_S: VRAM ヒット 60〜77% → 68〜89%、+12〜15%） |
| `STRATA_GLM_BG_SPLIT` | Qwen3.8 は 1、他は 0 | CPU の道が RAM 層のエキスパートの一部だけを受け持つときも、背景の入れ替えを動かす（0: 全部を受け持つときだけ。以前のまま）。Qwen3.8 で、VRAM の枠にエキスパートの半分以上が入るときは ratio 1 / margin 2 で入れ替える（RTX 5090、Swift 1.5 IQ2_XS 32 GB: VRAM ヒット 95.0 → 96.2%、176 → 182 tok/s）。枠が少ないときは 2 / 8 のまま |
| `STRATA_GLM_FETCH_COLS` / `STRATA_GLM_CPU_WAIT_COLS` | Qwen3.8 は 1、他は 0 | 先読みを確かめる窓で、CPU の答えを待つ処理とエキスパートを取ってくる処理を、列ごとではなく 1 回のカーネルで全部の列にする（計算は同じ） |
| `STRATA_GLM_GRAPH` / `STRATA_GLM_GRAPH_MOE` | CUDA は 1 / 1、HIP は 0 | Qwen3.8 の確かめる窓で、delta-net の層を CUDA graph で流す（0: 1 つずつ起動。`_MOE=0`: MoE の振り分けは graph の外）。出力は同じ。1 トークンの起動回数 453 → 254（RTX 5090: 速さは変わらない。DGX Spark で PDL と合わせて: 43.4 → 47.1 tok/s） |
| `STRATA_GLM_PDL` / `STRATA_GLM_PDL_PREFETCH` | 内蔵 GPU（sm_90 以上）と、外付けの sm_120 以上のカード（RTX 50）の Qwen3.8（v1.3.132）と Inkling 2（この版から）は有効。ほかの外付けのカードとモデルは無効 / 1 | programmatic dependent launch: 前のカーネルが終わりかけたところで次を始める（DGX Spark、Qwen3.8 16G64G: MTP なし 25.0 → 30.7 tok/s、MTP あり 43.4 → 46.5。RTX 5090、Qwen3.8 UD 3.1、4K + 512: MTP なし 128.0 → 137.8、MTP あり 178.6 → 184.7。Inkling 2 Mini UD-Q4_K_XL、短い文脈、token の graph あり: 150.2 → 162.8。GLM-5.3 / V4 / V4.1 は RTX 5090 で速くならなかったので無効のまま）。`_PREFETCH`: GEMV が重みの行を早めに L2 へ読む（0: 読まない。2: PDL なしでも読む） |
| `STRATA_GLM_UMA` / `STRATA_GLM_UMA_HEADROOM_GB` | 1 / 8（CUDA） | 内蔵 GPU（DGX Spark）は共有メモリから層の大きさを決め、`_HEADROOM_GB` を空けておく（0: 外付けのカードと同じ決め方。比べる用） |
| `STRATA_GLM_UMA_SHARE` / `STRATA_GLM_UMA_OTHERS_GB` | 0 / 16 | 内蔵 GPU で、ほかのプロセスが共有メモリを `_OTHERS_GB` より多く使っていて、そのせいでエキスパートが入りきらないときは、読み込みの段階で止まり、メモリを多く使っているプロセスの名前を出す（エンジンを 2 つ動かしたままにして、DGX Spark のメモリが尽きたため）。`STRATA_GLM_UMA_SHARE=1` で、止めずに動かす。ほかに何も動いていないときに、機械より大きいモデルを動かす場合は止めない |
| `STRATA_GLM_UMA_DIRECT` | 0 | 内蔵 GPU で、RAM 層のエキスパートを VRAM の枠に写さず、その場で読む。既定は無効: 入れ替えが止まり、ヒット率が低いままになる（DGX Spark、Qwen3.8 16G64G 32 GB: 写す方式 45.8、その場で読む方式 34.9 tok/s） |
| `STRATA_GLM_DISK_BUFFERED` | 0 | （試験用）SSD からのエキスパートの読み込みを、O_DIRECT ではなくページキャッシュ経由にする |
| `STRATA_GLM_BG_PACE_KB` | 1024。CUDA の Qwen3.8 は 512 | tier の入れ替えのコピー（prompt の後の補充、入れ替え）を、token の境目でまとめて出さず、MoE 層ごとに最大この KB ずつ、次の層の mixer の間に流す（v1.3.28）。まとめて出すと、次の token の host-mapped の読みがコピーの終わりまで待たされていた（8K prompt の後、約 10 token が 35〜39 ms）。CUDA の Qwen3.8（エキスパート約 2 MB）は v1.3.132 から 512：1 MiB ずつでも次の層のルートの書き込みが待たされていた（RTX 5090、UD-IQ3_XXS: 110 → 118 tok/s）。GLM-5.3 / V4 / V4.1（エキスパート 7〜8 MB）は 1024 のまま（512 は速くならず、減衰カウントの効果を消した）。`0`: 従来どおり境目で |
| `STRATA_GLM_BG_DECAY` / `STRATA_GLM_BG_DECAY_N` / `STRATA_GLM_BG_DECAY_MARGIN` | Qwen3.8、GLM-5.3、DeepSeek V4 / V4.1、Hy4 の CUDA は 0.995 / 8 / 2（HIP は Qwen3.8 だけ）。ほかは 0 | バックグラウンドの tier の入れ替え（v1.3.132）で、エキスパートの順位を、半分にしていく整数のカウントではなく、token の境目ごとにこの割合で減っていくルートのカウント（約 200 token 分の記憶）で決めます。RAM 層のエキスパートのカウントが、その層で VRAM にいる一番冷たいものを `_MARGIN` 上回ったら入れ替え、1 回の境目で最大 `_N` 個。重みを動かすだけなので答えは同じです（決定論モードの md5 は変わりません）。RTX 5090、4K + 512、iGPU なし：Qwen3.8 UD 3.1 UD-IQ3_XXS 128.5 -> 136.1 tok/s（MTP 185.5 -> 196.6）、DeepSeek V4 0731 UD-Q2_K_XL 53.7 -> 58.2、GLM-5.3 Maya-S-v2 IQ2_XXS 43.8 -> 46.0、V4.1 Compact JA256 47.3 -> 49.0、Hy4（mix459 + 512、RAM 層 60 GB）自前 32G96G 46.8 -> 50.8、Maya-XS IQ2_XXS 57.1 -> 59.9（40 GB: 47.9 -> 52.7、56.9 -> 59.6）。`0`：以前のカウントと `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN` |
| `STRATA_GLM_BG_LANE` | GLM-5.3、DeepSeek V4 / V4.1 の CUDA は `512:128:1`。ほかは off | `<レーンのエキスパート 1 個あたり KiB>:<引く KiB>[:<種類>[:<最大 KiB>]]`。CPU レーンが route の RAM 層のエキスパートを ne 個受け持つとき、サービススレッドが待ちのある tier のコピー（種類 1 = 昇格、2 = 降格、3 = 両方）を ne × KiB − 引く KiB だけ、2 本目の copy stream ですぐ流す。CPU が計算している間、リンクは空いていた（V4 0731 の nsys：1 層あたり約 215 µs、96% 空き）ので、その間に渡り、GPU が答えを読む前に終わる。重みを動かすだけで答えは変わらない（決定論 md5、移動ありの厳密レーンの md5 とも同じ）。RTX 5090 + Gen4 x4、4K + 512、iGPU なし：V4 0731 UD-Q2_K_XL 58.3 → 60.7 tok/s、GLM-5.3 Maya-S-v2 IQ2_XXS 46.8 → 49.1、V4.1 Compact JA256 50.0 → 52.2（5 組の計測で +4〜8%）。Qwen3.8 には使わない（レーンの 1 個が約 30 µs で空きが無く −7.5%）。`0`：off |
| `STRATA_Q4X_GRAPH_PDL` | 外付けの CUDA カードは有効。内蔵 GPU と HIP は無効 | Qwen3.8 の検証窓の層ごとの graph（`STRATA_GLM_GRAPH`）を、PDL が有効なとき PDL の辺つきで capture します（ドライバが断ったら辺なしで取り直し）。RTX 5090 + MTP：同じ token、同じ速さ（190.1 / 190.0 tok/s）。`0`：辺なし（v1.3.132 より前と同じ） |
| `STRATA_Q4X_GDN_MID` / `STRATA_Q4X_GDN_MID_PF` | 1 / 0（CUDA の sm_90 以上。HIP は使いません） | Qwen3.8 のデコード：delta-net の層の alpha / beta、conv + SiLU、再帰を、qkv / z の GEMV の後の 1 回の起動で計算します。同じ計算を同じ順で行うので、同じビットです（`gdn_core_parity`）。`_PF=1`：再帰の間に `ssm_out` の重みを L2 へ読みます（既定は無効。UD 3.1 で無効 122.7 tok/s、有効 120.5。状態の読み出しとぶつかります）。`STRATA_Q4X_GDN_MID=0`：別々の起動 |
| `STRATA_GLM_ZERO_COPY` | オン | copy engine を使わない token の境目（v1.3.21）: 表の更新・埋め込み行（`token_embd` の pinned コピー、Q8_0 / Q5_K）・トークンを、カーネルが host-mapped で直接読み書きする。`0`: 従来どおり memcpy |
| `STRATA_GLM_GAP_PROF` | オフ | `1`: 答えごとに `glm gap prof` の行を出す。トークンとトークンの間の時間の内訳（host 側のラップと、timing event で測った GPU の空き） |
| `STRATA_GLM_TAIL_SKIP` / `_W` | オフ | ルートの r 位より下（または確率の割合が閾値より下）で VRAM にないエキスパートを計算しない。CPU レーンが遅い環境で速くなるが、答えが変わる（KL 0.01〜0.02 程度） |
| `STRATA_GLM_ROUTE_AFFINITY` | オフ | `<a>`: VRAM にあるエキスパートの選択スコアに +a する（重みはルーターの値のまま）。僅差なら GPU にあるエキスパートが選ばれる（Maya #252 から。v1.3.36）。答えが変わる。GLM-5.3-Flash Maya-S-v2、RTX 5090 で `0.04` が 1 トークン 14.1 ms（既定 15.4、−9%）、VRAM のヒット 56 → 69%、nll 1.631（既定 1.635、既定のロジットに対する KL 0.043）。`0.08` は −13% だが nll が悪化し始める（1.641〜1.651）。hash ルーティングの層は対象外 |
| `STRATA_GLM_ROUTE_AFFINITY_RAM` | オフ | `<a>`: RAM 層のエキスパートにも同じく +a（VRAM にあるものは大きい方）。SSD の層がある構成で、SSD にしかないエキスパートを避ける（v1.3.37 で計測）。DeepSeek V4.1 の tail skip 設定、8 本の評価用シーケンスで、`0.08` が 1 トークン −8〜12%（SSD の読み −22%）、KL 0.2406 → 0.2438、ppl 9.76 → 9.81。`0.16` は −12〜20%、KL 0.2501 / ppl 9.89 |
| `STRATA_GLM_ROUTE_PROF` | オフ | `1`: route kernel が 43,000 ルートごとに段ごとの SM サイクルを出力する（診断用、NVIDIA） |
| `STRATA_GLM_TAIL_SKIP_DISK` | オフ | `1`: 上の省略を、SSD にしかないエキスパートだけに限る（v1.3.29）。`STRATA_GLM_TAIL_SKIP=3` と合わせて DeepSeek V4.1 の速度設定: 1 トークン 56.6 → 34.2 ms、KL 0.229 → 0.240。`=4`（5〜6 位だけ）なら、AI PRO R9700 + レーンの UD-IQ2_XXS で 1 トークン 48.5〜53.4 → 39.8 ms、KL +0.007（`=3` は 38.6 ms、+0.020。2026-10-27）で、こちらの方が釣り合いが良い。全エキスパートが VRAM + RAM に載るなら何も変わらない |
| `STRATA_GLM_RAM_FILL` | `STRATA_GLM_TAIL_SKIP_DISK=1` のとき 2、それ以外は 0 | トークンごとに、SSD にしかないエキスパートのうちルートに選ばれた回数が多いものを、この数まで背景で RAM 層に読む（issue #8: SSD 限定の省略は、飛ばしたエキスパートを計算しないので読まれることがなく、ずっと SSD に残って飛ばされ続けていた。会話を通して 1 トークン 16〜25 個）。RAM 層のより使われないエキスパートより回数が `STRATA_GLM_RAM_FILL_MARGIN`（4）以上多いときは、その場所に読む。`0`: オフ。エンジンのログに回数が出る（`glm fast tail skip ... RAM fill`） |
| `STRATA_GLM_DEFER` | オフ | expert deferral: 層の MoE のうち CPU レーン側の分を層の終わりで待たず、次の層の attention の後に（元の層の重みで）residual に足す。GPU がレーンを待たなくなる: RTX 5090 + Radeon 8060S で単発 1 トークン 10.9 → 9.5 ms、2 トークンの検証窓 14.4 → 13.1、tok/s +12〜15%。答えの変化は tail skip より大きい（Q8_0 参照に対する KL 0.08、オフなら 0.012。swe-bench L7 16 問 × 3: 107.2 tok/s、resolved 16 問中 13。オフなら 100.9、14）。層分割でも各 part の中で遅らせる（part の最後の層は自分の分を待つ）。AI PRO R9700 で iGPU レーンなし（v1.3.41、DeepSeek V4）: 1 トークンずつ 46.1〜46.5 → 53.0〜53.2 tok/s、CUDA 版の基準に対する KL 0.059（オフなら 0.0057。1 位のトークンの一致はオフの 97% に対して 90%） |
| `STRATA_DS4_IDX`, `STRATA_DS4_ATTN` | 0 | `1`: lightning indexer / prompt の attention を tensor core 版ではなく従来の FP32 カーネルで計算する（A/B 確認用）。v1.3.149 から DeepSeek V4.1 の 32 ヘッドの indexer も tensor core で計算する（CUDA sm_80 以上：生成・確認ウィンドウ・prompt、HIP gfx11 / gfx12：生成。HIP の prompt は以前から WMMA）。変わるのは和の順序だけ（トークン経路に対する KL は FP32 カーネルと同じで、実行ごとのぶれの範囲）。RTX 5090 + 123 GB、iGPU なし、V4.1 Compact JA256、v1.3.148 との中央値：prompt 32K / 64K / 128K が 1,488 / 2,127 / 2,564 → 1,549 / 2,425 / 3,252 tok/s（+4% / +14% / +27%。128K の prompt の indexer の GPU 時間 11.3 → 1.2 秒）、その後の生成 63.6 / 63.8 / 61.3 → 64.9 / 65.8 / 62.3 tok/s（+2.0% / +3.2% / +1.7%）。128K 行で 1 回 123 → 23 µs。`1` で以前と完全に同じ出力 |
| `STRATA_DS4_PATTN` | 0 | `1`: prompt の attention を、1 ブロック 32 ヘッドではなく 16 ヘッドの tensor core 版（attend_batch_mma_kernel）で計算する（v1.3.151。CUDA sm_80 以上で 1 ブロック 90 KB の共有メモリが取れる GPU：キーのタイルを 32 ヘッドで 1 回だけ cp.async で集め、前のタイルの計算と重ねる。softmax はレジスタの中）。出力はビット単位で同じ。RTX 5090、2,048 トークン × 640 キーの 1 回 3.38 → 2.27 ms。V4.1 JA256 の prompt 32K / 64K / 128K で +4.3% / +2.2% / +2.7%、V4 0731 の 128K で +4.2%（A/B 確認用） |
| `STRATA_DSA_TOPK_MB` | 1 | DeepSeek V4 / V4.1 の生成と確認ウィンドウで、indexer の top-k の選択（と V4.1 の候補ブロック）を、選ぶ元が 12,288 行以上のとき複数の GPU ブロックで行う（選ぶ数が 1,024 を超えるときは大きさによらず）。選ばれる行も順番も同じで、答えは変わらない。RTX 5090 で 32K 行 20.6 → 11.2 µs、128K 行 49.5 → 11.3 µs、V4.1 の 2,048 個の候補ブロックは 95〜137 → 13〜34 µs。Radeon 8060S で 32K 行 25.2 → 10.6 µs。エンジンの中（RTX 5090、128K の prompt）では、1 トークンあたりの選択の時間が V4 0731 で 0.49 → 0.32 ms（生成は誤差の範囲で 70.8 / 70.6 tok/s）、V4.1 JA256 で 0.48 → 0.14 ms（60.5 → 61.8 tok/s。64K は 63.3 → 64.4）。`0`：常に 1 ブロック。`N`（2 以上）：N 行から。`STRATA_DSA_TOPK_MB_SLICE`（1 ブロックあたりの行数、1024）は調整用 |
| `STRATA_Q35_ATTN` | 0 | `1`：Ornith（qwen35moe）の生成 / 確認ウィンドウの attention を従来のカーネル（v1.3.64）に戻す。0：分割カーネル（v1.3.65）。8 キー分を 1 回のシャッフルで合計し、softmax は 32 キーのタイルごと、K / V は共有メモリ経由。115K の位置で 1 層 1 トークン 1.09 → 0.57 ms、3 行のウィンドウ 3.29 → 1.39 ms。`STRATA_Q35_ATTN_PW`（warp あたりの query-head の組、2）と `STRATA_Q35_ATTN_CHUNK` / `_CHUNK_MIN`（ブロックあたりのキー数。既定は文脈の約 1/128、最小 64）は調整用 |
| `STRATA_Q35_PATTN` | 0 | `1`：Ornith（qwen35moe）のプロンプトの attention を従来の FP32 カーネル（v1.3.66）に戻す。0：テンソルコア（rocWMMA / CUDA の WMMA。FP16 の入力、FP32 の合計。v1.3.67）。114K の位置での 1,024 トークンのサブバッチが 1 層 918 → 156 ms、115K のプロンプトが 685 → 154 秒。24K の文脈の後の KL は FP32 カーネルに対して 0.0054（FP32 同士の 2 回で 0.0064） |
| `STRATA_PROMPT_FA_OLD` | 0 | `1`: prompt の attention を以前のカーネルに戻す（Laguna は CUDA コアのカーネル、qwen35 系 （Ornith-35B / 122B、Qwen3.5-122B、Qwen3.8-27B）は WMMA / FP32 のカーネル。以前と完全に同じ結果）。0: これらを 1 つの tensor core カーネルで計算する（CUDA sm_80 以上、FP16 のキャッシュ。HIP と q4_0 / q8_0 の KV キャッシュは以前のカーネルのまま）：mma.sync、softmax はレジスタの中、キーを 1 回だけ読む（WMMA 版は 2 回）、window 層のリングはそのまま読む。q と P は FP16、合計は FP32 で、和の順序が変わる：32K の文脈の後の llama.cpp に対する KL は Laguna 0.0246 → 0.0248、トークン経路に対する KL は Ornith-35B 0.0105 → 0.0088、Qwen3.8-27B 0.0039 → 0.0030、Ornith-122B 0.0040 → 0.0044。RTX 5090 + 123 GB、iGPU なし、prompt 60K / 125K の tok/s：Laguna-S-2.1 UD-IQ4_XS 530 / 259 → 4,108 / 3,403、Ornith-1.5-35B-A3B Q4_K_M 1,907 / 911 → 8,281 / 6,043、Qwen3.8-27B UD-Q4_K_XL 1,893 / 1,262 → 2,667 / 2,131、Ornith-1.5-122B 24G64G 2,198 / 1,492 → 2,927 / 2,356。生成の速さは同じ（出力の文が同じとき） |
| `STRATA_PROMPT_MMQ` | 0 | `1`: prompt のエキスパートを、どの重みの型でも llama.cpp の MMQ カーネルで計算する（v1.3.160 と完全に同じ結果）。0: RTX 5090 クラスの GPU（sm_120）では、IQ2_XXS、IQ2_XS、IQ3_XXS、IQ2_S、IQ1_S、IQ4_NL、Q2_0、IQ4_XS、Q4_K、Q5_K、Q6_K のエキスパートの積を Abyss 自前の int8 tensor core カーネルで計算する（mma.sync、gate/up の書き戻しで SwiGLU、活性の丸めは MMQ と同じ）。足す順番が変わる：token 経路に対する KL は GLM-5.3-Flash 0.0200 → 0.0205、V4 0731 0.0053 → 0.0055、V4.1 0.0082 → 0.0071、Qwen3.8 0.0237 → 0.0243（回ごとのぶれ 0.001）。prompt は GLM / V4 で 4K〜128K が +0.1〜1.8%、V4.1 は 32K で最大 +10%。dense（qwen35）の層（FFN とその隣の射影）は、Q6_K を除く同じ型と Q4_0 を自前で計算する（Qwen3.8-27B UD-IQ2_XXS / UD-IQ3_XXS で +2〜8%、UD-Q4_K_XL で +0.9〜2.5%、L-Q4_0 で +6〜12%。Ornith-35B は +0.6〜3.2%、Ornith-122B は同じ速さ。dense の Q6_K は MMQ より遅いので MMQ のまま）。ほかの GPU、ほかの型（MXFP4、NVFP4、IQ3_S、Q3_K、Q2_K、Q4_1、Q5_0、Q8_0）、HIP は MMQ のまま。 |
| `STRATA_PROMPT_OWN_TYPES` | 未設定 | 自前の prompt カーネルに渡す重みの型を、上の既定の一覧の代わりに指定する。`all`（読める型すべて：MMQ が読む型全部と IQ1_M。Q8_0、dense の Q6_K、MXFP4、NVFP4 は MMQ より遅い。IQ3_S、Q3_K、Q2_K、Q4_1、Q5_0 はモデルでは未計測）か、GGUF の型番号をカンマ区切りで（例 `16,18`）。sm_120 以外の GPU、HIP（gfx11 / gfx12 の WMMA。今は MMQ より遅い）でも有効になる。`STRATA_PROMPT_MMQ=1` の方が優先。 |
| `STRATA_Q35D_MMQ` | on | Qwen3.8-27B（dense の qwen35）のプロンプト: dense FFN の積を量子化したままの重みで MMQ で計算する（4K プロンプトで FFN 1,187 → 563 ms）。`0`: FP16 に展開して cuBLAS。HIP で未設定のときは、最初のプロンプトで 1 層の gate の積を両方で計り、MMQ が 1.5 倍より遅ければ MMQ をやめる（ログに `the dense FFN's MMQ is slow on this GPU / driver` の行。#49: Windows の RX 9060 XT では MMQ で SenseNova-Loop-9B のプロンプトが 3 tok/s、Radeon 8060S は MMQ のまま 575 tok/s、FP16 だと 141）。`1` は確認なしで MMQ を使う。`STRATA_Q35D_MMQ_CHECK_SCALE`（テスト用）は計った MMQ の時間を何倍かにする。`STRATA_Q35D_MMQ_ROWS`（1 回に通すトークン数、2048: 約 330 MB）は調整用 |
| `STRATA_Q35D_SPLIT` | オン | 約 24 GB 未満の GPU の Qwen3.8-27B（v1.3.140）: 各層の FFN の先頭の行を VRAM に置き、GPU と CPU が自分の行を並行して計算する。`0`: 行ではなく層まるごとを CPU に置く（A/B 用: 16 GB・下書きなしで 42.2 に対し 22.2 tok/s） |
| `STRATA_GLM_PREFILL_PROF` | オフ | `1`: プロンプトのたびに `glm prefill: N tokens so far, GPU stages X ms a token ... | host waits on disk .., plan .. ms a token` の行をログに出す（プロンプト経路の段階別の時間）。`--bench` のレポートにも載る（#43） |
| `STRATA_GLM_Q35_PROJ_MMQ` | オフ | `1`: qwen35moe（Ornith、Qwen3.5-122B、Ornith-35B、Laguna）のプロンプトの mixer の射影（GDN の qkv / z / ssm_out、attention の q / k / v / out）を、展開 + hipBLAS / cuBLAS の代わりに量子化したままの重みで MMQ に通す（バッファ 18 MB。RTX 5090 で 0.076 → 0.057 ms/トークン）。足す順序が変わるので出力が既定と数トークン目から分かれる。そのため既定はオフ |
| `STRATA_GLM_MMQ_GLU` | オン | prompt の expert の gate/up の積で SwiGLU まで済ませる（NVIDIA Turing 以降）。`0` で別カーネル（以前のビルドとビット一致） |
| `STRATA_GLM_CPU_MT` | オン | 複数トークンが同じ expert を使うとき（prompt、検証窓）、CPU は重みの行を 1 回だけ展開する。ggml の内積とビット一致（IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS は `STRATA_GLM_CPU_1T` の CPU で格子をレジスタの中で引く。v1.3.41）。`0` で無効 |
| `STRATA_GLM_CPU_1T` | オン | CPU レーンの 1 トークン用の行を軽くする（ggml の内積とビット一致）。VNNI と VBMI のある AVX-512 の CPU（Zen 4 / Ice Lake 以降）では、IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS の格子をレジスタの中で引く（v1.3.41。Ryzen AI Max+ 395 の 16 スレッドで DeepSeek V4 のエキスパート 1 個が 109 → 90 µs、内蔵 GPU なしの生成が +3〜9%。`STRATA_NO_IQ512_VBMI=1` でこれだけ無効）。それ以外の CPU では AVX2 の IQ3_S / IQ3_XXS。`0` で無効 |
| `STRATA_CPU_GGML` | オフ（自前の内積に SIMD 版がない aarch64 で、ggml 入りのビルドならオン） | `1` で、CPU のエキスパートが内積と活性の量子化に以前の ggml-cpu を使う（`STRATA_GGML=ON` でビルドしたときだけ）。`0` で aarch64 でも自前のものを使う。自前の内積は同じビットを出す（own_dot_test が全型を ggml-cpu と照合） |
| `STRATA_GGML_BUILD` | 未設定（Linux x86-64 は ggml なし、Windows・aarch64・HIP は ggml あり） | ビルド時に `./abyss.sh build` / setup が読む。`off` で `-DSTRATA_GGML=OFF`（自前の CPU の内積と GGUF の型表、ggml のライブラリなし。Windows・aarch64・HIP では beta なので報告歓迎）、`on` で ggml のライブラリを使う。GPU のプロンプトの MMQ は、まだ llama.cpp のソースから作る |
| `STRATA_GLM_VRAM_EVICT` | 使用回数 | `lru`: VRAM 層の入れ替えで、最も長く使われていない expert を追い出す（キャッシュの調査用） |
| `STRATA_GLM_DISK_READERS` | オフ | prompt 処理で SSD 上の expert をこの数のスレッドで読む（DRAM なしの NVMe 向け。環境ごとに測ること。DeepSeek V4.1 は `8` で 1K トークンの prompt が 33 → 27 秒） |
| `STRATA_GLM_DISK_PAR` | オフ | `1`: 答えている最中、トークンが VRAM と RAM で見つけられなかったエキスパートを、専用のスレッドで SSD から読む。スレッドはエキスパート 1 個の片の数だけ（3 部 × `STRATA_GLM_READ_CHUNKS`（8）＝ 24。呼び出したスレッドを含む）で、1 個の片を全部同時に読む（今は 9 スレッドで約 3 巡）。`<n>`（2〜64）でスレッド数を指定。変わるのは読む順だけで、答えは同じ。スレッドは読みの中で待ち、読みと読みの間は `STRATA_GLM_DISK_PAR_SPIN_US`（50）µs 回ってから眠る。`STRATA_GLM_TIMING=1` のとき、オンでもオフでもディスク読みの統計（待っている間の速さ、エキスパート 1 個の待ちの平均 / p50 / p90 / p99）を出すので、比べるのに使える。まだ測っていない（fio、6.7 GB/s の NVMe：424 KB の読みを 9 本同時で 4.64 GB/s、24 本同時で 5.10 GB/s） |
| `STRATA_GLM_MIRROR` | オフ | `<フォルダ>[:<フォルダ>...]`: 別の SSD に置いたモデルのシャードの写し（同じ名前）。エキスパートを読むたびに、早く終わりそうな方の SSD から読む（答えている最中と起動時の読み込みの両方）。DeepSeek V4.1 UD-IQ2_XXS、AI PRO R9700、2 枚目は USB4: SSD 待ち −20%、32 GB + 64 GB で +21%（v1.3.77、[SSD2](SSD2.ja.md)）。Linux のみ |
| `STRATA_GLM_DISK_LANE` | オフ | `1`: SSD から RAM に読んだエキスパートを、VRAM に送らずその場で CPU / iGPU レーンが計算する（v1.3.29。V4.1 の Q2_K/Q3_K エキスパートで −5%、小さい Q2 では効果なし） |
| `STRATA_DS41_ENGRAM_CACHE_MB` / `_THREADS` | 256 / 15 | DeepSeek V4.1: Engram の行キャッシュと、行を読むスレッド数（v1.3.29） |
| `STRATA_DS41_ENGRAM_ASYNC` | オン | DeepSeek V4.1: トークンの Engram 48 行を、層 0 の計算中に裏のスレッドが読み、層 1 / 14 が GPU 上で到着を待つ（v1.3.36。1 トークン −0.5 ms、出力は同じ）。`0`: トークンの前に読む |
| `STRATA_DS41_ENGRAM_WINDOW_ASYNC` | オン | DeepSeek V4.1: 検証窓の Engram 行も同じように裏で読み、各表を最初に使う層が GPU 上で到着を待つ（v1.3.117。RTX 5090 で窓の Engram の待ち 0.39 → 0 ms、出力は同じ）。`0`: 窓の前に読む |
| `STRATA_DS41_ENGRAM_PREFETCH` | 8 | DeepSeek V4.1: prompt の chunk の hash の直後に、その Engram 行を行キャッシュへ読むスレッド数（v1.3.36。1K トークンの prompt で Engram 1.7〜1.9 → 0.6 秒、prompt 全体 −6〜7%）。`0`: オフ |
| `STRATA_DS41_ENGRAM_STORE`, `STRATA_DS41_ENGRAM_STORE_THREADS` | 1, 64 | DeepSeek V4.1: prompt の chunk の Engram 行を、行番号が決まった直後から専用のスレッド（64）で chunk ごとの行ストアへ、2 つの Engram 層が使う順に読む。続けて次の chunk の分も読む（v1.3.151。以前は層が 256 トークンのブロックごとに大半の行を読み、その間 GPU が待っていた。128K の prompt で 11.4 秒）。読む行も出力も同じ。RAM: 2 chunk 分の生の行（32K トークンの chunk で約 0.45 GB）。RTX 5090 + 123 GB、iGPU なし、V4.1 JA256、STRATA_DS4_PATTN のカーネル込み、v1.3.150 との中央値：prompt 32K / 64K / 128K が 1,508 / 2,296 / 3,304 → 2,590 / 3,535 / 4,265 tok/s（+72% / +54% / +29%）、生成は変わらない。`STRATA_DS41_ENGRAM_STORE=0`: 以前の行キャッシュへの先読み（STRATA_DS41_ENGRAM_PREFETCH）とブロックごとの読み |
| `STRATA_DS41_ENGRAM_AHEAD` | オン | DeepSeek V4.1: 先読みスレッドが、今の chunk の後に次の prompt の chunk の Engram 行も読みます（JA256 4K で 2 つ目の chunk の層 1 が 425 → 74 ms。出力は同じ。v1.3.144）。`0`: 各 chunk の行だけ |
| `STRATA_DS41_ENGRAM_KEEPAWAKE` | 1000 | DeepSeek V4.1、Linux: 生成のトークンが来ている間、1 本のスレッドが 1 つ目の Engram 表の 4 KiB をこの間隔（us）ごとに O_DIRECT で読み、トークンとトークンの行の読みの間（約 15 ms）に SSD が遅い状態へ落ちないようにする（NVMe は 20 ms 休むと 4 KiB の読みが 3〜4 倍遅い：p50 140 → 440 us）。最後のトークンから 200 ms で止まる。RTX 5090 + Lexar NM790、V4.1 JA256、p4k + 512 トークン：1 つ目の表の行を待つ時間 836〜1,038 → 616〜807 us/トークン（平均）、生成の中央値 64.0 → 65.4 tok/s（+2%）。読む行も出力も同じ。生成中に約 4 MB/s の読みが増える。`0`: オフ |
| `STRATA_DS4_SPEC` | 1（オン） | prompt と応答から作る投機デコード。1 回に試す候補トークン数（1〜3。実測では 1 が最速）。`0` でオフ |
| `STRATA_DS4_SPEC_GATE` | オン | リクエストの採用率が検証窓の分に見合わない間は候補を出さない。`0`: 常に出す |
| `STRATA_DS4_SPEC_ADAPT` | DeepSeek V4 0731 と V4.1 1017 ではオン（`STRATA_GLM_WINDOW` をオンにした GLM-5.3-Flash も）、ほかのモデルではオフ | ドラフトモデルなしで、1 回の窓に最大 5 個の n-gram の下書きを出す。各位置は当たる割合が保たれている間だけ使い、幅は `STRATA_DS4_SPEC_COST` で決める（v1.3.145。DSpark の生成の繰り返しの部分だけ）。1 回の窓に 1 トークンとゲートの代わり。V4 0731: 書き換え / swe-bench の文 +20.9 / +11.9%、散文 +0.5 / −1.7%。V4.1 JA256 は pipelined の窓（v1.3.146）で書き換え / swe-bench の文 +28.8 / +19.8%、散文 +1.0 / −0.7%。`0` / `1` で固定 |
| `STRATA_DS4_SPEC_GATE_FRESH` | `STRATA_DS4_SPEC_ADAPT` がオンのとき（V4 0731、V4.1 1017）オン、ほかはオフ | ゲートとコスト表の時間に、応答の最初の 16 ステップを入れない（prompt の直後はプールが prompt に貸した expert を取り戻すので 1 トークンが遅い。V4.1 JA256 で 21.3 ms、あとは 17.3 ms。窓が安く見えていた）。窓が続くときは 32 ステップに 1 回 1 トークンだけで進めて、その時間を測り直す（v1.3.145）。`0`: すべてのステップを数える |
| `STRATA_DS4_SPEC_MIN` | 3 | 候補を探すときに一致させるトークン数（2〜4） |
| `STRATA_DS4_SPEC_COST` | 繰り返しだけの生成（`STRATA_DS4_SPEC_ADAPT`）ではオン、ドラフトモデルありではオフ（v1.3.145） | 窓の大きさを、決まった規則ではなく、この機械での幅ごとのかかる時間と下書きが残る割合から決める（v1.3.117 で作り直し: 表はプロセスのあいだ持ち続け、幅ごとの時間は直近 16 回の中央値、1 トークンだけの時間もときどき測り直す）。`0` / `1` で固定 |
| `STRATA_DS4_SPEC_COST_MARGIN` | 1.10（DeepSeek V4 0731 は 1.02） | `STRATA_DS4_SPEC_COST` で、窓の 1 ミリ秒あたりのトークン数が 1 トークンだけのときをこの倍率で上回るときだけ窓を使う（v1.3.145）。窓の列は 1 トークンなら使わない expert も呼ぶので、そのあとのステップで VRAM に当たる割合が下がる（GLM-5.3-Flash の swe-bench の prompt で VRAM ヒット 56.7 → 45.6%）。1.02 では、窓自身の時間だけ見れば五分の窓のせいで GLM-5.3-Flash と V4.1 JA256 がそこで 4〜5% 遅くなった。1.0〜2.0 |
| `STRATA_DS4_NGRAM_LONG` | オフ | `1`: n-gram の下書きを、前に出てきたいちばん長い一致から出し、一致が長いほど多く下書きする（n-gram だけの生成。v1.3.117）。Qwen3.8 の写す作業で +8%、DeepSeek V4 0731 では窓が増えて −12% なので既定はオフ |
| `STRATA_GLM_SAMPLE_COLS` | オン | 温度が 0 より大きいとき、検証窓の列を、列ごとのコピー・起動・待ちではなく、1 回の起動と 1 回の待ちでまとめて引く（v1.3.117。出力は同じ。`STRATA_GLM_SAMPLE_COLS_CHECK=1` で各列を前のやり方でも引いて食い違いを数える）。`0`: 1 列ずつ |
| `STRATA_GLM_WINDOW` | VRAM が 24 GB 以上の NVIDIA のカードでは `2`（`STRATA_GLM_VRAM_GB` があればその値、なければカードのメモリで判断。v1.3.167）、共有メモリ（DGX Spark）では `1`、24 GB 未満と AMD ではオフ | GLM-5.3-Flash を GPU 1 枚で：検算の窓つきの投機的デコード。出力は窓なしと同じ（v1.3.118）。`2`: NextN 層を読まずに窓だけ使い、`STRATA_DS4_SPEC_ADAPT` の n-gram の下書きを pipelined の窓（`STRATA_GLM_WINDOW_PIPE`）で確かめる。RTX 5090 + RAM 層、Maya-S-v2 IQ2_XXS で `0` と比べて（書き換え / swe-bench の文 / 散文 p4k / 散文 rail）：32 GB で +28.5 / +11.9 / −0.2 / −2.2%（rail は 3 回の幅の中）、24 GB（`STRATA_GLM_VRAM_GB=24`）で +15.7 / +3.8 / +0.7 / +0.5%。16 GB では swe-bench の文が −5.3%（窓のバッファが expert の置き場を取る）なので、オフにしている。`1`: モデル自身の NextN 層が最大 `STRATA_Q35_MTP_K`（3）トークンの下書きを作り、1 つの窓でまとめて確かめる。DGX Spark：94 GB の S1 で 21.4 → 24.3 tok/s、unsloth の UD-IQ2_XXS で 19.8 → 22.9。単体の GPU + RAM 層では同じか遅い。`0`: オフ（v1.3.167 より前の、外付けのカードの既定）。検査（stderr の数、すべて 0 であること）: `STRATA_GLM_WINDOW_CHECK=1/2`、`STRATA_GLM_REWIND_CHECK=1`、`STRATA_GLM_MVN_CHECK=1` |
| `STRATA_GLM_WINDOW_PIPE` | `1` | GLM-5.3-Flash の検算の窓（`STRATA_GLM_WINDOW`）を V4.1（v1.3.146）と同じ pipelined にする。列を 2 組に分けて別のストリームで流し、片方の組の注意を GPU が計算する間に、もう片方の組の RAM 層のエキスパートを CPU レーンが計算する。出力はまとめて流す窓と同じ。`0`: 1 本のストリームでまとめて流す（v1.3.145〜v1.3.166 の、手で選ぶ `STRATA_GLM_WINDOW=2` の窓。RTX 5090 で書き換え +9.3%、swe-bench の文 −1.8〜−2.8%） |
| `STRATA_GLM_DENSE_Q8` | オン | GLM-5.3-Flash：GGUF の Q8_0 の密な重み（attention の k_b / v_b、indexer、KDA の f/g/beta、hc）を BF16 に広げずに Q8_0 のまま読む。1 トークンで読む量が 0.37 GB 減り、KL は 0.2708 → 0.2667（v1.3.118。DGX Spark で +2%）。`0`：前の BF16 のコピー |
| `STRATA_GLM_KDA_SPLIT` | オフ | `1`：1 トークンの KDA の再帰を、1 ヘッド 4 つの列のまとまりで計算する（出力は同じ。GB10 でも RTX 5090 でも速くならなかった） |
| `STRATA_GLM_COLS_ORDER` | オン（CUDA） | 検算の窓で列をまとめて走らせるエキスパートの計算（GLM-5.3-Flash、DeepSeek V4 / V4.1、DSpark の下書き）で、グリッドの先頭を列にする。2 つの列が同じエキスパートを使うとき、2 回目はメモリではなく L2 から読む（v1.3.119。同じブロックを順番だけ変えて走らせるので、出力は同じ）。DGX Spark で GLM の S1 も DeepSeek V4 0731 + DSpark も速くなった（CHANGELOG を参照）。`0`：前の順番。AMD 版は前の順番のまま |
| `STRATA_GLM_MLA_COLS` | オン | GLM-5.3-Flash の検算の窓：attention の k_b / v_b の GEMV を、全部の列で 1 回ずつまとめて走らせ、重みを 1 回だけ読む（v1.3.119。どの列の和も前と同じで、出力は同じ）。`0`：1 列ずつ |
| `STRATA_GLM_KDA_OOP` | オン | GLM-5.3-Flash の検算の窓：1 列目が KDA の状態を巻き戻し用の控えへ直接書き、次の列はそこから読む。巻き戻すときも同じように控えから計算し直す。窓ごとに 146 MB × 2 の状態のコピーがなくなる（v1.3.119。和は同じ）。`0`：コピーする |
| `STRATA_GLM_ROUTE_COLS` | オン（Qwen3.8） | 検算の窓で、列ごとのルーティングを 1 回の起動でまとめて行う（v1.3.119。同じ計画を同じ順番で作る。列ごとに起動すると窓 1 つで 48 × T 回、DGX Spark で 1 回 ~13 us かかっていた）。`1`：ほかのモデルでも使う、`0`：列ごとに起動 |
| `STRATA_GLM_ROUTE_QUIET` | オン（Inkling） | `STRATA_GLM_ROUTE_COLS` のとき、どの列も共有の expert 状態を書き換えない（選んだ expert がすべて VRAM にあり、CPU lane も helper もない）起動では、列の順番待ちをせずに全列を同時にルーティングする。計画と要求は同じで、それ以外の起動は今までどおり順番に行う（Inkling `--parallel` 8 本: route 1 層 38.7 -> 18.4 us）。`1`：ほかのモデルでも使う、`0`：オフ |
| `STRATA_GLM_ROUTE_QUIET_MIN` | 4 | `STRATA_GLM_ROUTE_QUIET` を使う最小の列数（2 列では待ち合わせのほうが高くついた） |
| `STRATA_GLM_MVN_PDL` | オン | 検算の窓の密な GEMV（`mvn`）を PDL（programmatic dependent launch）で起動し、前の起動を待つ前に重みの行を L2 へ先読みする（1 トークンずつの GEMV と同じやり方。v1.3.119、CUDA。和は同じ）。`0`：前の起動のしかた |
| `STRATA_Q35_MTP_Q4` | オフ | `1`（試し）：Qwen3.8 の MTP の下書きが、Q8_0 の射影（IQ4_NL）と head（Q4_K）の 4 bit のコピーを読む（読み込み時に作る）。下書きは速くなるが、採用は減る（DGX Spark：87.6% → ~80%）。どのトークンも検算の窓が決めるので答えはモデルのものだが、使わないときとトークンが変わることがある（v1.3.119） |
| `STRATA_DETERMINISTIC` | オフ | `1`（確かめ用、遅くなる）: 時計の時間で決まる選択を止める。prompt の CPU / iGPU の分担、生成の CPU / iGPU レーン、裏でのエキスパートの移動と先読み、n-gram の判定、pack の `expert_usage.txt`。また、prompt のエキスパートの足し算を、プールの中身によらず同じ順にする（`STRATA_MMQ_WHOLE_K=1`、`STRATA_GLM_PREFILL_WINDOW=0`）。同じリクエストを 2 回走らせると同じトークンになる（出力を変えないはずの変更の A/B 用）。環境変数ですでに指定したものはそのまま。**確かめ用のスイッチで、ふだん使う設定ではありません:** エキスパートの置き場所（VRAM / RAM の層）を起動時のまま固定するので、そこにないエキスパートはトークンごとに毎回読み直します。PCIe が詰まりやすいカードでは大きく遅くなり、RTX 4090 + 96 GB で Qwen3.8 UD-Q4_K_XL は 54.1 から 14.6 tok/s（3.7 倍遅い）、RAM からの読み込みは 1 トークンあたり約 20 から 214 回に増えました（#39）。ふだんの使い方で毎回同じ答えにしたいときは、Qwen3.8 ではレーンの厳密な合算（`STRATA_GLM_LANE_XC`、v1.3.127/v1.3.128 から既定でオン）が速度を落とさずに使えます |
| `STRATA_DS4_SPEC_PIPE` | auto（HIP）、1（NVIDIA） | 検証の窓（DeepSeek V4。V4.1 も v1.3.146 から。それまでは batched だけ。RTX 5090、V4.1 JA256 の書き換えで batched 55.8 対 pipelined 73.8 tok/s）: `1` pipelined（v1.3.43 から列を 2 つのグループに分けて 2 本の HW キューに流し、グループの中はまとめて計算）、`0` batched（dense の重みを窓で 1 回だけ読む）、`auto`: 窓の幅ごとに、最初の窓で両方を 3 回ずつ測り、速いほうをその幅で使い続ける（v1.3.40。幅ごとは v1.3.43 から。HIP の既定。NVIDIA ではまだ確認していない）。RTX 5090 + iGPU では pipelined（swe-bench 90 対 73 tok/s）、AI PRO R9700（HW キュー 2 本、v1.3.43）ではどの幅も pipelined（2 / 4 / 6 / 8 列で 34〜36 / 52〜56 / 71 / 88 ms、batched は 41 / 68.5 / 98 / 133） |
| `STRATA_INK_DRAFT` | off | `1`（opt-in）: Inkling 2 Mini の verify window。今の token と最大 5 個の下書きを 48 回の層の適用にまとめて通す。どの列も token 1 個のときと同じ計算（dense の GEMV は `mvn_mv`、norm は 1 行ずつ、attention の split はその位置の token 単独と同じ、短い畳み込みの状態は読むだけで、受け入れた token の分だけ進める）なので、greedy の答えはあり・なしで同じ token になる。下書きは prompt lookup（書いている文章が前の部分の繰り返しのとき: 編集、コードの写し）、窓の幅は 1 ms あたりに残ると見込める token 数で決める（`STRATA_DS4_SPEC_COST`、GLM / V4 の decode と同じ）。RTX 5090、UD-Q4_K_XL: 短い文脈で編集タスク 131.5 → 150.2 tok/s（下書きの 64% を採用）、日本語 131.1 → 142.9、コード +1%、散文は変わらず。8K では編集 +3%、コード -2.5%。未設定なら token の経路は今まで通り |
| `STRATA_INK_DSPARK` | 自動 | Inkling 2 Mini の DSpark drafter（unsloth の `dspark-Inkling-2-Mini-Q8_0_Old.gguf`、402 MB、VRAM 0.6 GB。2026-11-23 より前にダウンロードした `dspark-Inkling-2-Mini-Q8_0.gguf` も同じ形で、そのまま読める。11-23 からは unsloth のリポジトリのこの名前のファイル（504 MB）が新しい形の drafter（llama.cpp 本流の dflash。`markov_w2` があり `dspark_fuse` が無い）に替わった。Abyss はまだこの形を読まない。エンジンはファイル名を挙げてそのことを 1 行出し、drafter なしで動き続ける）が verify window（`STRATA_INK_DRAFT` のもの。これを設定すると一緒に有効になる）に 1 ステップ最大 7 個の下書きを出す。本体の 5 / 23 / 41 回目の層の入力と最後の norm 後の行を読む小さな 3 層で、1 ステップに 8 行のブロックを 1 回、出力ヘッドは本体と共有、2 個目以降の下書きは Markov ステップ。prompt lookup の繰り返しが見つかればそちらを先に使い、窓の幅はコスト表と下書きの確信度で決める。残る token は本体が出したものなので、greedy の答えは同じ token（md5 で確認）。RTX 5090、UD-Q4_K_XL、512 token で 3 回の中央値、token と窓の graph（`STRATA_INK_GRAPH`、`STRATA_INK_WIN_GRAPH`）とゲート（`STRATA_INK_DSPARK_GATE`）あり: コード 185.2 → 223.4 tok/s、編集 182.0 → 241.2、8K のコード 182.7 → 236.8、8K の編集 182.8 → 244.5。165K の prompt のあとの 1,024 token は散文 97.4 → 108.7、コード 97.2 → 117〜158。日本語の本文は 181.1 → 237.6（1,024 token、決定モード）。少し遅いのは英語の思考と散文だけで、512 token で 179〜183 → 176〜181（token 約 5.2 ms に対して下書き約 1.4 ms と、ゲートの測定の分）。未設定のとき（次の版から）: セットアップが探す場所（`../Abyss-data/models/unsloth/Inkling-2-Mini-GGUF/dspark/`、または pack のフォルダの近く。どの場所でも `_Old` の名前を先に見る。abliterated 版の pack も unsloth のフォルダを見つける。setup はダウンロードしない）にファイルがあれば、エンジンが自分で読み込み、drafter があっても expert のプールが同じ（層あたりのスロット数と、カードに余裕があった場合のプロンプトの尾や予備のスロット。`STRATA_INK_DSPARK_POOL_LOSS` でスロットの減りを一定の割合まで許せる。試す人向け）なら残す。そのうえでゲート（`STRATA_INK_DSPARK_GATE`）が、下書きのほうが速いと測れたところだけ下書きする。そうでなければ 1 行出して外す（`ink dspark: off - its 0.66 GB would come out of the expert pool (...)`）。RTX 3090（24 GB）、UD-Q4_K_XL: `--max-context 65536`（setup の値）と `131072` は 128 スロットのままで残る。`196608`（119 → 113）と `262144`（102 → 96）は外す。`STRATA_DETERMINISTIC=1` のとき（tier が固定で CPU レーンもないので、窓の各列が RAM 側の expert を PCIe で取りに行く。RTX 5090 の 165K のプロンプトで 85.0 → 62.0 tok/s）、`--parallel` が 2 以上、層の分割、unified memory、HIP では自動では読み込まない。`0`（`off` / `none`）: 使わない。`<ファイル>`: VRAM にかかわらず読み込む（これまでどおり）。`./setup-deepseek.sh --model inkling2mini --no-dspark` は `0` を書く。setup が見つけたファイルを env に書いてある設定（v1.3.195〜v1.3.199 の setup が書いた）は、setup をもう一度走らせるとその行が消え、この確認が効く |
| `STRATA_INK_DSPARK_N` | 7 | drafter が 1 ステップに出す下書きの数（1〜7。窓は今の token と最大 7 個の下書きを確かめ、幅はコスト表で決める） |
| `STRATA_INK_DSPARK_CTX` | 32768 | drafter の global attention 層が context の K/V を何位置分持つか（context 長で頭打ち。他の 2 層は 1,024 位置の窓） |
| `STRATA_INK_DSPARK_DEFER` | 1 | 1 token ずつ進んだ位置の drafter の文脈の行を、accumulator にためておき（最大 64 行）、次の下書き・verify window・プロンプトの chunk の前にまとめて 1 回で書き込む。token ごとに後ろで走っていた書き込み（小さい起動 約 12 本）の代わり。行も下書きも同じ。ゲートが drafter を休ませている間（散文・日本語）は書き込みがほぼなくなる: RTX 5090、決定モードの散文 175.1 -> 176.3 tok/s。`0`: token ごとに書き込む |
| `STRATA_INK_DSPARK_GATE` | 1 | drafter のゲート: 下書きを出した 16 ステップごとに、その token 数 ÷ 時間（下書き + token か窓）を 1 token ずつの速さと比べる。下書きが負けたら drafter を 32、64 … 512 ステップ休ませ（1 token ずつ。prompt lookup の繰り返しは試す）、また測る。1 token ずつの速さは下書きなしのステップで測る（`STRATA_INK_DSPARK_PROBE`）。Inkling の token は CUDA graph 1 つ（RTX 5090 で約 5.4 ms）なので、下書き 1 回の約 1.1 ms は散文・日本語では元が取れない。残る token はどちらでも本体のもの。`0`: 毎ステップ下書きする |
| `STRATA_INK_DSPARK_GATE_EARLY` | 4 | ゲートの早めの負け判定: 測定中に下書きをこの回数出した時点（16 未満）で、速さが 1 token ずつの 0.85 倍を下回ったら負けとして測定を打ち切る。散文・日本語で drafter が早く休む。`0`: 常に 16 回まで測る |
| `STRATA_INK_DSPARK_GATE_FIRST` | 0 | 答えの最初のこの数のステップは drafter を使わない（1 token ずつ）。そのあとでゲートが測り始める（試験用） |
| `STRATA_INK_DSPARK_GATE_AFTER0` | 0 | 下書きを出して 1 つも使わなかったステップの後、この数のステップ drafter を休ませる（試験用） |
| `STRATA_INK_DSPARK_PROBE` | 256 | ゲートが比べる 1 token ずつの速さ: 答えの最初の下書きなしのステップ（`STRATA_INK_DSPARK_PROBE_FIRST`）、続く下書き 16 回のあとに下書きなしの 8 ステップ、そのあとは下書きこの回数ごと、負けて休むあいだのステップでも測る（直近 8 ステップの中央値）。窓のあいだに挟まる 1 token のステップは窓が呼んだ expert の中で動くので遅い。24 GB のプールで 32K のプロンプトのあと、下書きなしの 7.9 ms に対して 16.5 ms かかり、下書きなし 127 tok/s に対して 97 tok/s の下書きが勝ちと判定されていた。`0`: これまでの基準（窓のあいだの 1 token） |
| `STRATA_INK_DSPARK_PROBE_FIRST` | 6 | 答えの最初のこの数のステップは drafter を使わない。ゲートは最初の下書きの前に 1 token ずつの速さを知るので、早めの負け判定が最初の下書き 4 回から効き、コスト表にまだない幅の窓は 1 token の値段から見積もる（2 列目から 1 列ごとに 1.2 + 0.3 倍）。#58、RTX 3090、250K のプロンプト、`STRATA_INK_DSPARK_POOL_LOSS=8`、2 回: 256 token の答え 41.4〜43.5 → 43.2〜45.2 tok/s（off 40.4〜40.6）、1,536 token の答え 47.7〜47.9 → 47.7〜50.3。短いプロンプトは回ごとの揺れの範囲。token は同じ（Inkling 18 md5、RTX 5090 と RTX 3090）。`0`: 最初のプローブは下書き 8 回のあと（v1.3.204 と同じ） |
| `STRATA_INK_DSPARK_POOL_LOSS` | 0 | 試す人向け（`STRATA_INK_DSPARK` が未設定のとき）: drafter の VRAM で減る層あたりの expert スロットがこの % 以下なら読み込んだままにする（`0`: プールが変わらないときだけ。v1.3.200 と同じ）。`8` で測定（#58、2〜3 回ずつ）: RTX 3090 の `--max-context 262144`（102 → 96 スロット）と、24 / 20 / 16 GB に絞った RTX 5090（262144 で 101 → 94、65536 で 117 → 112 と 83 → 77）で残る。コード、編集、日本語、8K のプロンプト、思考ありの答え、128K / 250K のプロンプトは速くなる（3090 でコード 93 → 118〜126 tok/s、約 3K token の思考ありの答え 94 → 133〜143。24 GB の 5090 で 250K のプロンプトのあとの散文 74 → 88〜93）が、散文は 3〜6% 遅く（スロットが減る分だけで 2〜4%）、短いプロンプトが 3090 で 0.15〜0.25 s 長くかかる（プールの分）ので、既定にはしない |
| `STRATA_INK_DSPARK_GATE_LOG` | 未設定 | `1`: ゲートの測定が終わるたびに、位置・ステップ数・1 token ずつと比べた速さをログに出す。`2`: 1 ステップずつも出す（種類・下書き数・token・ms） |
| `STRATA_INK_CTL_BF16` | 1（CUDA）、0（HIP） | Inkling 2 Mini: ループの制御器の行（unsloth の F32 の `loop_*` の down / gain / forget、144 テンソル）を GPU 上で BF16 で持つ。中の値はどれも BF16 で表せる値（下位 16 bit が 0。読み込み時にテンソルごとに確かめ、そうでない値が 1 つでもあるテンソルは F32 のまま）なので、同じ値で bytes は半分。重みの領域の中でその場で詰める（VRAM の使用量は変わらない）。GEMV は F32 の GEMV と同じ積を同じ順に足す（結果は bit まで同じ。Inkling の md5 も KL の行も変わらない）。プロンプト経路は SGEMM の前に F32 に戻す。RTX 5090、UD-Q4_K_XL、server モード、3 回の中央値: 生成 189.7 → 193.0 tok/s、16K のプロンプトの後 201.5 → 204.6。16K のプロンプトは変わらず（8,366 / 8,360）。`0`: F32 の行のまま。AMD は `1` のときだけ（AMD では測っていない） |
| `STRATA_INK_CTL_CHECK` | 未設定 | `1`: 読み込み時に、詰めた制御器のテンソルごとに GEMV（1 列と 3 列）を同じ乱数の入力で F32 のものと bit まで比べ、層ごとに 1 行ログに出す（試験用） |
| `STRATA_INK_CTL_SPLIT` | 未設定 | `2` / `4` / `8`（opt-in）: 制御器の down の GEMV（4,608 の行が 64 本）を 1 行あたりその数の warp で計算し、部分和を順に足す。足す順番が変わるので結果は bit まで同じではない。RTX 5090、UD-Q4_K_XL、server モード: `2` で生成 192.9 → 193.0 tok/s（速くならない）、`4` で 192.9 → 195.1。llama.cpp の Q8_0 参照に対する KL（3,071 位置）: exact 0.005687（日本語 0.0068、同じ top token 96.94%）、`2` 0.005636（0.0068、96.97%）、`4` 0.005681（0.0071、96.71%）。`4` は速いが日本語が少し悪いので、どちらも既定にしていない |
| `STRATA_INK_FUSE` | 1 | Inkling 2 Mini の decode: 1 層の小さい kernel をまとめる。どれも元の kernel と同じ計算（結果は bit まで同じ）。深さ方向の attention の mix とその後の norm（または制御器の入力の norm）、その前の後置きの畳み込みと残差の加算を 2 本に（chunk ごとの和、その後 1 行 1 block）。制御器の apply と norm を 1 本に。attention の結合と sigmoid の gate と q8_1 を 1 本に。k / v の短い畳み込み、q / k / rel の clip、KV キャッシュへの書き込みを 1 本に。RTX 5090、UD-Q4_K_XL、短い文脈: `STRATA_INK_GRAPH`、`STRATA_GLM_PDL` と合わせて 131.8 → 185.6 tok/s（1 token の kernel 1,540 → 1,020 本）。`0`: 別々の kernel のまま（token の graph も使わない） |
| `STRATA_INK_MIX1` | 1 | Inkling 2 Mini の decode（token、verify window、DSpark の経路）: 深さ方向の attention の mix を、その前の後置きの畳み込みとその後の norm（または制御器の入力の norm）ごと、`STRATA_INK_FUSE` の 2 本から 1 本の launch に。chunk の block が和を作り、softmax と重み付きの和、最後の block が norm をやる（互いをカウンタで待つ）。どの和も今までと同じ順に足すので、結果は bit まで同じ。RTX 5090、UD-Q4_K_XL、server、3 回の中央値: prose 189.8 → 195.0 tok/s、code 189.7 → 195.1。128 行以上の prompt: mix を、その前の後置きの畳み込みと残差の足し算、その後の norm と FP16 の写しごと、1 行 1 block の 1 本の launch に（128 行未満は GPU を埋めやすい旧来の launch のまま）。結果は bit まで同じ。16K prompt prose 8343 → 9074 tok/s、code 8383 → 9083、2,048 行 6845 → 7278、512 行 3303 → 3386。`0`: どちらも旧来の launch のまま（HIP は常に旧来） |
| `STRATA_INK_GRAPH` | 1 | Inkling 2 Mini の decode: 1 token 分（48 回の層の適用と head の mix）を 1 本の CUDA graph で流す。同じ kernel を同じ順に（結果は bit まで同じ）。位置は device の 1 語から読む。graph は attention の split 数ごとに 1 本（それぞれ 1 回だけ記録、1 会話で多くて 70 本ほど）。プロファイルの印、dump、エキスパートの先読み、expert helper、画像の行があるときは使わない。`0`: kernel を 1 本ずつ起動 |
| `STRATA_INK_WIN_GRAPH` | 1 | Inkling 2 Mini の verify window（`STRATA_INK_DRAFT` / `STRATA_INK_DSPARK`）: 窓の 48 回の層の適用と head を、（幅, attention の split の格子）ごとに 1 本の CUDA graph で流す。同じ kernel を同じ順に（結果は bit まで同じ）。キーごとに 6 回目の窓の後で記録する（記録 1 回で約 13 ms）。split の格子は 2 のべき乗に切り上げ（最小 8）、長い文脈でも graph は少なくて済む。プロファイルの印、dump、エキスパートの先読み、expert helper、画像の行があるときは使わない。`0`: kernel を 1 本ずつ起動 |
| `STRATA_INK_WIN_CAP_AFTER` | 6 | graph のキーごとに、記録する前に 1 本ずつ起動する verify window の数（`STRATA_INK_WIN_GRAPH`。記録 1 回で約 13 ms） |
| `STRATA_INK_PREFILL_ATTN` | 1 | Inkling 2 Mini のプロンプト（CUDA、sm_80 以降）: プロンプト経路の attention をテンソルコアの kernel で（トークンのブロック x キーのタイル、FP16 の積を F32 で足す、相対バイアスは FP16 2 つに分けて F32 相当、窓の外のタイルは飛ばす）。attention のサブバッチは 2,048 行（`STRATA_INK_PREFILL_SUB` で変えられる）。RTX 5090、UD-Q4_K_XL で 8K プロンプト 3,784 → 6,874 tok/s、165K 765 → 5,298。足す順番と q / P の FP16 丸めでビットは変わるが、llama.cpp に対する KL は変わらない。`0`: トークン経路の kernel で 1 行ずつ、サブバッチ 512 行（前のビット）。AMD は常にこちら |
| `STRATA_LING_MTP` | オフ | Ling-3.2-air: GGUF の NextN ブロック（blk.42）が下書きし、verify window がモデルと突き合わせる。残るトークンはモデルのもの（greedy の md5 はオフと同じ。KDA 層の状態は window の前に写しを取り、残った列だけやり直す）。`1`: `<pack>/nextn.gguf`。一度 `.venv/bin/python tools/nextn_pack.py --gguf <1 本目のシャード> --out <pack>/nextn.gguf --experts q5_0` で作る（2.11 GB。bartowski の Q4_0 のエキスパートを同じ値のまま Q5_0 に詰め直す。ggml は不要）。`<path>`: そのファイル。window あたりの下書き数は `STRATA_Q35_MTP_K`（Ling は 1。RTX 5090 + RAM、bartowski の Q4_K_M、4K と 16K の後の code / prose / ja / edit を 3 回ずつ: decode はオフに対して 1 で +3〜+16%、2 や 3 では -5〜+15%）。プロンプトは 6〜8% 遅くなる（ブロックの行の分）。既定はオフ: 測ったのは 1 台だけ（エキスパートの読み出しの 66% が 5090 の上）で、ファイルを作る手間もある |
| `STRATA_LING_PROMPT_ATTN` | 1 | Ling-3.2-air のプロンプト（CUDA、sm_80 以降）: MLA 層のプロンプトの attention をテンソルコアのフラッシュ kernel で（FP16 の積を F32 で足す、オンライン softmax、ヘッドごとのゲートは途中で掛ける）。アキュムレータの配置を初回に確かめ、合わなければ戻る。`0`: 1 行ずつ（行ごとに decode の kernel） |
| `STRATA_INK_PARALLEL` / `--parallel` | 1 | Inkling 2 Mini、`--serve`: まとめて decode するリクエストの数（`N`、または `auto`: 全 expert が VRAM に残り、3 GB とプロンプト経路の領域が余る範囲で最大 8。それより大きい `N` はそこまで減らす）。実行設定の `"parallel"`。[複数のリクエストを同時に](#複数のリクエストを同時にparallelinkling-2-miniベータ) を参照 |
| `STRATA_PAR_CHUNK` | プロンプト経路の chunk | `"parallel"` が 2 以上: 他のリクエストが decode 中のとき、decode の合間に読むプロンプトのトークン数（`STRATA_PAR_PREFILL_MB` 2048 の 32 GB のカードで 1,280）。小さくすると走っているストリームの止まる時間が短くなり、プロンプトは遅くなる。また長いプロンプトの区切りがほかの decode の有無で変わるので、答えの最後のビットも変わりうる |
| `STRATA_PAR_YIELD` | 8 | `"parallel"` が 2 以上: 他のリクエストが decode 中に読むプロンプトの chunk の中で、48 層のこの数ごとにそれらの decode を 1 ステップ挟む（chunk の和は変わらない）。小さいほど止まりは短く、プロンプトは遅い（4: p99 23 ms、8K 1.84 s。12: 48 ms、1.31 s）。0: 無効（chunk 全部のあと `STRATA_PAR_SHARE`） |
| `STRATA_PAR_SHARE` | 0（`STRATA_PAR_YIELD=0` なら 0.5） | `"parallel"` が 2 以上: プロンプトの chunk のあと、その chunk にかかった時間のこの割合だけ走っているリクエストを decode してから次の chunk へ（0: chunk の間に decode 1 ステップ） |
| `STRATA_PAR_SNAP_ASYNC` | 1 | `"parallel"` が 2 以上: 読み終えたプロンプトの会話（その lane の次のリクエスト用）を、他のリクエストの decode 中にコピー用 stream でホストへ写す。そのリクエスト自身はコピーが終わってから始まる。0: decode の stream で（全ストリームが待つ。8K の会話で約 75 ms） |
| `STRATA_PAR_PREFILL_MB` | 2048 | `"parallel"` が 2 以上: プロンプト経路専用の VRAM 領域（MB）を起動時に確保し、expert pool の尾を借りない。借りると尾の expert が RAM へ行って戻り、その間ほかのリクエストが待った（40 トークンのプロンプトで約 150 ms）。0: 1 本のときと同じく借りる |
| `STRATA_PAR_TRACE` | 未設定 | `1`: プロンプトの chunk ごとに、各段階の時間（lane の切り替え、会話の復元かリセット、読み込み、snapshot）を 1 行出す |
| `STRATA_INK_FP4ACT` | 0 | Inkling 2 Mini のプロンプト、**丸めが変わる opt-in**（CUDA の sm_120 = RTX 50 シリーズだけ。既定のビルドにある sm_120a のコードを使う）: NVFP4 の専門家（層 0〜19。Q8_0 の層 20〜23 は MMQ のまま）を、活性も FP4 にしてブロックスケールの FP4 テンソルコアで。gate/up は専門家のスケールと SwiGLU まで 1 本の積で。`1`: 値ごとに FP4 1 項（16 値ごとの UE4M3 スケールの下の E2M1、行ごとに float のスケール 1 つ）。`2`: 2 項（1 項目の残差を同じ形で量子化、x ≈ q1 + q2、約 8 bit）。既定の `0` は今までの exact な int8 経路（bit まで同じ）。RTX 5090、UD-Q4_K_XL、server、16K プロンプト、6 回の中央値: prose 9,075 → 13,122 tok/s（`1`、+45%）/ 11,798（`2`、+30%）、code 9,064 → 13,117 / 11,792。プロンプト経路の KL（llama.cpp の Q8_0 参照、3,071 位置。括弧内は日本語の 1/3）: exact 0.0056（0.0059）、`1` 0.0141（0.0144）、`2` 0.0059（0.0077）。top-1 は 96.9% / 95.5% / 96.7%。decode、verify window、DSpark は変わらない。ほか（HIP、ほかの GPU）は stderr に 1 行出して exact な経路 |
| `STRATA_DS4_SPEC_GROUP` | 窓の半分 | pipelined の窓の 1 グループの列数（v1.3.43）。既定は ceil(T / 2) 列ずつの 2 グループ |
| `STRATA_DS4_WIN_COLS` | オン | 窓のグループで、VRAM にある expert の gate/up と down を列ごとではなく全列まとめて 1 回ずつ起動する（v1.3.43。ビット一致: `STRATA_DS4_WIN_COLS_CHECK=1` で 16 グループに 1 回、列ごとの起動と照合）。`0`: 列ごと |
| `STRATA_GLM_CPU_MULTI` | オン | batched の窓で、1 層ぶんの CPU レーンの expert を全列まとめて 1 回で計算し、重みの行は全列で 1 回だけ読む（v1.3.43。ビット一致）。`0`: 列ごと |

コンテキスト長は `--max-context` で決める。DeepSeek V4 の学習は 1M トークンだが、既定は 65536 で、131072（125K の
prompt）まで測った。アテンションのキャッシュは小さい（65536 で 0.44 GB）が、prompt 用のバッファはコンテキストに
比例して増える。

## llama.cpp との照合

`tools/dsv4_logits_dump.cpp` は、トークン列に対する llama.cpp の logits を書き出す（`deepseek4` に対応した llama.cpp に
対してビルドする）。エンジンの `STRATA_GLM_SCORE` モードで、エンジン自身の logits と比べられる。手順と結果は
[docs/deepseek4/PORT.md](deepseek4/PORT.md)。UD-Q2_K_XL では、3000 トークンで KL 0.009、次トークンの的中率も同じだった。

## うまくいかないとき

- **ビルドが `mma with block scale not supported on .target sm_120` で止まる** - `-DCMAKE_CUDA_ARCHITECTURES=120a` で
  構成し直す。
- **`the engine exited before it was ready`** - エンジンのログ（設定の `log`）を見る。RAM 層の確保でメモリが足りないなら、
  `STRATA_GLM_RAM_GB` を下げるか `STRATA_GLM_RAM_HEADROOM_GB` を上げる。
- **`refers to missing files` や `pack:` のエラー** - pack は、作ったときの 3 つのシャードと同じ場所に置く必要がある。
  シャードを移したら作り直す。
- **思考が一言で止まって回答が始まる** - `max_tokens` が小さすぎる（上の「思考」を参照）。
- **遅く、SSD がずっと動いている** - エキスパートに対して RAM + VRAM が足りない。ほかのプログラムを閉じるか、RAM を増やす。
- **ポートが使われている** - 別のサーバーが動いている。`--port` で変える。

## 制限

- MTP のドラフトはない。この GGUF には MTP ブロックがない。DeepSeek の DSpark ドラフトモデルは v1.3.43 から使える
  （`--dspark`、GPU 1 枚。[DSpark のドラフトモデル](#dspark-のドラフトモデルdeepseek-v4-flash)）。使わないときの投機デコードは、
  文章自体から候補を作る（`STRATA_DS4_SPEC`）。
- 会話スロット（`STRATA_GLM_SLOTS`）は DeepSeek V4 では無効。直前の会話の再利用は動く。
- 画像は扱わない（ここでは DeepSeek V4 Flash はテキストのみ）。
- 複数 GPU: 層の分割は動く（「複数の GPU」）が、遅い 2 枚目を足しても速い 1 枚 + iGPU レーンには
  届かない。そのとき `auto` は速いカード 1 枚で動かす。Windows は DeepSeek V4 では未確認。
- `./abyss.sh`（Maya から引き継いだインストーラ）は GLM 用。DeepSeek V4 は `./setup-deepseek.sh`。DeepSeek V4.1 は
  `./setup-deepseek.sh --model v41`（zephel01 の LDLQ GGUF）か手作業（[docs/deepseek41/README.md](deepseek41/README.md)）。
- DeepSeek V4.1: モデル自身の MTP 層でのドラフトはない（MTP 層を持つ GGUF がない。v1.3.53 からの DSpark ドラフター（任意）が
  代わり）。層の分割、会話のディスク保存はまだない。prompt のアテンションは v1.3.60 からまとめて計算する。v1.3.31 までは、
  コンテキストの 16,385 トークン目で落ちた（8K を超えるとアテンションの行選びも崩れていた）。長い会話には v1.3.32 以降を使う。
