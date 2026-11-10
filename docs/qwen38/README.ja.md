# Qwen3.8-Flash-Next を Abyss で動かす

English: [README.md](README.md)

Qwen3.8-Flash-Next（llama.cpp の `qwen4exp`）は、v1.3.74 から Abyss の速い経路（`--glm-pack`）で動きます。プロンプト
経路、検証窓、MTP ドラフターも使えます。古いエンジン経路（[docs/maya/ORCA.md](../maya/ORCA.md)）もそのまま動きますが、
速い経路のほうがプロンプトは数倍、答えも速くなります。

計測は orcarouter の
[Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) IQ3_XXS
（2 シャード、85.2 GB）で、AI PRO R9700（32 GB、PCIe Gen4 x4）、RAM 94 GB（`STRATA_GLM_RAM_GB=60`）、iGPU レーンありの
環境です。NVIDIA でも動きます: RTX 5090 は v1.3.83 から、DGX Spark は v1.3.113 から（ベータ）。数字は
[NVIDIA での計測](#nvidia-での計測rtx-5090dgx-spark)にあります。

## モデルの形

- **48 層**: delta-net 層が 36（出力ゲートは sigmoid）、4 層ごとに注意層が 12（クエリ 24 ヘッド × 256、KV 2 ヘッド、
  q / k の norm、RoPE 64 次元、sigmoid の出力ゲート）。
- **残差は 4 本の流れ**（各 2560）で、低ランクの hyper-connection が混ぜます（norm のテンソルは無く、混ぜる処理が norm を
  兼ねます）。
- **層 1 の per-layer embedding（PLE）**: 直前のトークンの n-gram のハッシュで、28.8 GB の表から 16 行を引きます。表は SSD
  に置いたまま直接読みます（1 トークン約 1 ms）。
- **QSA**（注意層）: 2,051 セルを超えると、索引が選んだ 4 セル × 512 ブロックだけに注意します（2,051 セルまでは全部に注意）。
- **1 層 512 エキスパート、10 個を選択**、加えて sigmoid ゲート付きの共有エキスパート。幅はどれも 640。
- GGUF には MTP（下書き）ブロックが無いので、ドラフターは別ファイルです（下で説明）。

## 準備

1. **pack**（密な重みを GGUF のままの形式で。BF16 への変換はしません）:

   ```sh
   .venv/bin/python tools/iq_pack.py --fast \
     --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
     --out /path/to/IQ3_XXS/pack-fast
   ```

   2 つのシャードがどちらもダウンロードし終わっている必要があります。pack は約 300 MB で、エキスパートと PLE の表は GGUF
   から直接読みます。

2. **ドラフター（任意、おすすめ）**: 古い経路と同じ MTP の実行用ファイルを、同じ手順で作ります
   （[docs/maya/ORCA.md](../maya/ORCA.md#preparation): `tools/mtp_fetch.py`、`tools/mtp_pack.py --experts q2_0`、
   `tools/mtp_rt.py`、最後に `data/draft_vocab.bin` をコピー）。できたディレクトリには `dense.bin`、`dense.txt`、
   `experts.bin`、`draft_vocab.bin`（計約 820 MB）が入ります。エンジンはこれを VRAM に置きます（キャッシュ込みで約 1 GB）。

3. **設定ファイル**（`abyss-<名前>.json`。[docs/SETTINGS.ja.md](../SETTINGS.ja.md) を参照）:

   ```json
   {
    "exe": "build-hip/strata",
    "args": ["--glm-pack", "/path/to/IQ3_XXS/pack-fast", "--max-context", "32768", "--prefill", "auto"],
    "tokenizer": "/path/to/IQ3_XXS/pack-fast/tokenizer",
    "model_name": "qwen3.8-flash-next-uncensored-iq3_xxs",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "port": 8082,
    "backend": "hip",
    "env": {
     "STRATA_GLM_RAM_GB": "60",
     "STRATA_Q4X_MTP": "/path/to/mtp/rt"
    }
   }
   ```

   NVIDIA では `"exe": "build/strata"`、`"backend": "cuda"` にします（ほかは同じ）。

   起動は `.venv/bin/python serve/server.py --engine strata --config abyss-<名前>.json --port 8082` です（実際に使われる
   ポートはコマンドラインの `--port` です）。

## 設定項目

| 変数 | 既定 | 内容 |
| --- | --- | --- |
| `STRATA_Q4X_MTP=<ディレクトリ>` | なし | ドラフターの実行用ファイル。無いときは、繰り返し（n-gram）からだけ下書きを作ります |
| `STRATA_Q35_MTP_K` | 4（このモデル） | 1 回に作る下書きの最大数（1〜5） |
| `STRATA_Q35_MTP_MIN` / `STRATA_DS4_DSPARK_THR` | 0.75（このモデル） | この確信度以上の下書きだけを検証窓に入れます |
| `STRATA_Q4X_MTP_FULLHEAD=1` | オフ | ドラフターの head を、`draft_vocab.bin` の 10.6 万トークンではなく全トークンにします |
| `STRATA_Q4X_DENSE=1` | オフ | デバッグ用: QSA の選択を使わず、すべてのセルに注意します |
| `STRATA_Q4X_PLE_SYNC=1` | オフ | デバッグ用: 検証窓の PLE の行を、層 0 と並行ではなく層 0 の前に読みます |
| `STRATA_GLM_NO_MTP=1` | オフ | `STRATA_Q4X_MTP` があってもドラフターを切ります |
| `STRATA_Q4X_DFLASH=<gguf>` | オフ | v1.3.125: MTP 層の代わりに DFlash 2 のブロックドラフター（nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF、Q8_0）を使います。コードと一覧は速く、文章は遅くなります（[MANUAL](../MANUAL.ja.md#qwen38-flash-next)）。GPU 1 枚のときだけ |
| `STRATA_Q4X_DFLASH_N` / `STRATA_Q4X_DFLASH_PMIN` | 5 / 0.9 | 1 回の下書きの上限 / 下書きに必要な確信度 |
| `STRATA_GLM_PRIOR` | サーバーが設定 | v1.3.86 から、pack に自分のプロファイルが無いとき、サーバーは Abyss に入っているこのモデル用のルーティングのプロファイル（`data/expert_counts/qwen38-flash-next.txt`）を指します。エキスパートはその順に温まり、層ごとの VRAM の割り当てもルーティングに合わせます |

v1.3.112 の検証窓の CUDA グラフとまとめた待ちには、それぞれのスイッチがあります（`STRATA_GLM_GRAPH`、
`STRATA_GLM_GRAPH_MOE`、`STRATA_GLM_CPU_WAIT_COLS`、`STRATA_GLM_FETCH_COLS`、`STRATA_GLM_BG_SPLIT`）。
[MANUAL](../MANUAL.ja.md) を見てください。

## 計測（R9700、2026-10-28）

| | 古い経路（v1.3.73） | 速い経路（v1.3.74） |
| --- | ---: | ---: |
| プロンプト 30K トークン | 217 tok/s | 1,037 tok/s |
| 答え、文章（日本語 / 英語） | 52 tok/s | 64 / 65 tok/s |
| 答え、コードを書く | 66〜85 tok/s | 86 tok/s |
| 答え、コードのコピー | | 100〜107 tok/s |
| 答え、ドラフターなし | | 51〜53 tok/s |
| 31K トークンのプロンプトの合言葉 | 正解 | 正解 |
| 会話の次のターン（10.8K トークン） | 再利用 | 再利用: 10,802 トークン、0.28 秒 |

文章とコードの数字は、コマンドラインで貪欲生成した 300 トークンの答えのものです。サーバーの答えは、iGPU レーンや
エキスパートの置き場所の違いで回ごとに少し変わります。下書きの採用率は文章で 74%、コードで 86% です。検証窓 1 回は
30 ms、1 トークンだけなら 18.6 ms です。

品質: 5,600 トークンより後の 494 位置で llama.cpp と比べると、トークン経路で KL 0.026 / top-1 92.3%、プロンプト経路で
0.035 / 90.9% でした。このモデルでは llama.cpp 自身の HIP と CPU の差が KL 0.045 です。貪欲生成では、`STRATA_DETERMINISTIC=1`
のとき、下書き（MTP でも DFlash 2 でも）があっても無くても同じトークンが出ます。`tools/spec_exact_check.py` で確かめられます
（コード、一覧、日本語、バグ修正、4K のプロンプトを、それぞれ 512 トークン 3 回）。ふだんのモードでは、CPU レーンとエキスパートの
置き場所が時間で決まるので、下書きの有無にかかわらず、僅差のところで 2 回の出力が分かれることがあります。

## NVIDIA での計測（RTX 5090、DGX Spark）

RTX 5090（PCIe Gen4 x4）+ RAM 94 GB、MTP ドラフターあり:

| ファイル | Abyss | 結果 |
| --- | --- | --- |
| ISTA-DASLab GSQ-RCO IQ3_S、コンテキスト 32K | v1.3.83 | ドラフターあり 100.4 tok/s、なし 72.0。数学はどちらも 43 / 50 |
| | v1.3.86 / v1.3.87 | `--bench` の生成 116.1 / 128.5〜129.5 tok/s。bench/ref との KL 0.0022 |
| | v1.3.88 | プロンプト 8K 2,240〜2,376 tok/s、32K 3,002〜3,027 tok/s |
| zephel01 の 16G64G 版 | v1.3.112 | 4K プロンプト、512 トークン（3 回の中央値）: 32 GB でドラフターあり 131.3 tok/s、なし 90.0。16 GB で 96.4 / 71.4 |

DGX Spark（GB10、共有メモリ 128 GB、レンタル）、16G64G 版、v1.3.113: ドラフターありで 47.05 tok/s（CLI、4K、
512 トークン × 3 回の中央値。なしで 31.2）、サーバー経由で 43〜54 tok/s、8K のプロンプトは約 1,040 tok/s
（[docs/DGX-SPARK.ja.md](../DGX-SPARK.ja.md)）。

## ほかの GGUF（ベータ）

v1.3.76 から、速い経路で下の人気の量子化も使えます。どれも 1 台（上の R9700）で、同じファイルの llama.cpp との
対数確率の差と、短い速度の測定だけを確かめています。**対応はベータです**。数字がどうであっても、`./abyss.sh --bench`
を回して、結果を [bench の issue](https://github.com/zephel01/Abyss/issues/new?template=bench.md) に貼ってください
（ファイルのフォルダ名も添えてください）。これらのファイルでは、`--bench` の結果にもそのお願いが出ます。

| ファイル | サイズ | 常駐分（PLE 表以外） | Q8_0 との差（KL / 1 位一致） | Abyss と llama.cpp の差（KL） | 速度（R9700） |
| --- | ---: | ---: | --- | --- | --- |
| [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) IQ3_XXS | 75.8 GB | 47.0 GB | 0.398 / 71.6% | 0.019-0.022 | 文章 66 / コード 84（16 GB + 64 GB: 61 / 77） |
| ISTA-DASLab GSQ-RCO IQ3_S | 83.6 GB | 54.8 GB | 0.282 / 76.5% | 0.026 | 文章 51 / コード 77（16 GB + 64 GB: 45〜57 / 54） |
| [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-IQ4_XS | 93.7 GB | 64.9 GB | 0.191 / 80.4% | 0.038-0.043 | 文章 57 / コード 74（24 GB + 64 GB: 55 / 68） |
| unsloth UD-Q4_K_XL | 111.3 GB | 82.5 GB | 0.100 / 86.4% | 0.029-0.037 | 文章 55 / コード 71 |
| （比較用）orcarouter Uncensored IQ3_XXS | 85.2 GB | | | 0.026（5,600 トークンの実行） | 上の表 |

「Q8_0 との差」は、llama.cpp 自身で unsloth の Q8_0 と比べた KL です（1,024 トークン × 8 区間、小さいほど良い）。
「Abyss と llama.cpp の差」は同じファイルでの比較で、量子化ではなく移植の誤差を見ています。

「速度」は貪欲生成で 300 トークンの答え（文章: 英語の作文、コード: LRU キャッシュ）、MTP ドラフターと iGPU レーンあり、
R9700 まるごとでの値です。かっこ内の GPU + RAM は `STRATA_GLM_VRAM_GB` と `STRATA_GLM_RAM_GB=46` で制限した値です。
回ごとに数 tok/s ぶれます。

選び方（v1.3.76、R9700、unsloth の UD 3.1 より前。新しいおすすめは[いまの選び方](#いまの選び方)）: 上から順に見て、常駐分が「GPU のメモリ + エンジンが使える RAM（RAM のおよそ 75%、`STRATA_GLM_RAM_GB`）」に
収まる最初のファイルを選んでください。PLE 表（28.8 GB）は SSD に置いたままなので、ファイル全体が収まる必要はありません。

| GPU + RAM | ファイル | 常駐分 |
| --- | --- | ---: |
| 32 GB + 96 GB 以上 | unsloth UD-Q4_K_XL | 82.5 GB |
| 24 GB + 64 GB 以上 | **unsloth UD-IQ4_XS**: ここで測った中で品質がいちばん良く、収まるならまずこれ | 64.9 GB |
| 16 GB + 64 GB | ISTA IQ3_S（ファイル 83.6 GB） | 54.8 GB |
| それより少ない（12 GB + 64 GB、16 GB + 48 GB） | ISTA IQ3_XXS（ファイル 75.8 GB） | 47.0 GB |

一段大きいファイルでも動きます（収まらない分を毎トークン SSD から読む）が、遅くなります。`--bench` で、1 トークンが
SSD の読み込みをどれだけ待っているかがわかります。

これらのファイルのためにエンジンへ足したもの: エキスパートの Q2_0 / Q5_1 / Q8_0、検証窓の密な重みの IQ4_XS / Q5_1 /
Q2_0、プロンプト経路の Q5_1、BF16 の hyper-connection・`ssm_alpha` / `ssm_beta`・PLE の key / value（ISTA はこれらを
BF16 で持っています）。

### NVFP4（NVIDIA の FP4 チェックポイント、v1.3.101、ベータ）

[magnitudedev の NVFP4 GGUF](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF)（rev `4591036`、
3 分割で 127.8 GB）: エキスパートが NVFP4（FP4 の値、16 個ごとの FP8 の倍率、エキスパートごとにもう 1 つの倍率）、ほかは
Q8_0 で、PLE 表も Q8_0 です（54.4 GB、SSD から読む。常駐分 73.4 GB）。pack は上と同じ手順で作ります
（`tools/iq_pack.py --fast --gguf ...-00001-of-00003.gguf`）。RTX 5090 + RAM 96 GB（+ 8060S の iGPU レーン）、MTP
ドラフターありで確認しました:

| 確認 | 結果 |
| --- | --- |
| Abyss と llama.cpp の差（KL、英日 640 トークン） | 1 トークンずつ 0.015、プロンプト経路 0.018 |
| プロンプト処理（needle は見つかる） | 8K: 1,054 tok/s、32K: 2,515 tok/s |
| swe-bench 3 問 | 105.3 tok/s |
| llmbench の難しめ 10 問 | 8/10（2 問は長さの上限で打ち切り）、品質 76.1 |
| 日本語 3 本、7 往復のツール | 自然な日本語。テストは通り、最後の報告も日本語 |

プロンプト経路は Blackwell の FP4 テンソルコア（RTX 50、RTX PRO Blackwell、DGX Spark）で NVFP4 を計算します。ほかの
NVIDIA の GPU と AMD はまだ確認していません。ほかの GPU での報告も歓迎です（上と同じベンチの issue）。

### unsloth UD-IQ1_S（UD 3.1、v1.3.112、ベータ）

[unsloth の UD 3.1 UD-IQ1_S](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)（66.4 GB）は、1 つの層だけ gate と up
のエキスパートの型が違います（層 18: gate が IQ2_S、up が IQ2_XXS。down はすべて Q2_0）。v1.3.112 から、pack はその層を
`G,U` と書き、役割ごとにそれぞれの型で計算します。古いエンジンは理由を出して読み込みを止めます。pack は v1.3.112 の
`tools/iq_pack.py --fast` で作ってください（上と同じ手順）。iGPU レーンと別 GPU のヘルパーは、その 1 層を CPU に回します。
RTX 5090（PCIe Gen4 x4）+ RAM 94 GB、MTP ドラフターあり、iGPU レーンなしで確かめました:

| 確認 | 結果 |
| --- | --- |
| NVFP4 教師との KL（seq 0〜9、コンテキスト 2,048） | 0.2007、1 位一致 83.2%（同じファイルの llama.cpp: 0.2012 / 83.1%）。16G64G は 0.083 / 89.3% |
| 生成、4K プロンプト、512 トークン（3 回の中央値） | 32 GB: ドラフターあり 153.4 tok/s、なし 103.7。16 GB（`STRATA_GLM_VRAM_GB=16`）: ありで 78.0、8060S の iGPU レーンも使うと 109.6 |
| プロンプト | 4K 約 1,900 tok/s、10K（needle）2,734 tok/s |
| 実際の作業（bench/mgpu/run.sh、32 GB） | 日本語 OK、コード 4 / 5、8K needle 正解 |

ファイルは小さいぶん、品質は 16G64G より下がります。大きいファイルが入らないときに選んでください。

### unsloth UD-IQ2_XS（UD 3.1、v1.3.114、ベータ）

[unsloth の UD 3.1 UD-IQ2_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)（75.8 GB）は、gate と up の
エキスパートが IQ2_S / IQ2_XXS（同じ層の中では同じ型）、down が IQ4_NL / Q2_0 で、down の 1 層（層 4）だけが Q5_0 です。
v1.3.114 より前のエンジンは、読み込みのときに `unresolved weights: blk.4.expert types 22/22/6` と出して止まります。pack は
`tools/iq_pack.py --fast` で作ってください（上と同じ手順）。RTX 5090（PCIe Gen4 x4）+ RAM 94 GB（RAM ティア 50 GB）、
iGPU レーンなしで確かめました:

| 確認 | 結果 |
| --- | --- |
| NVFP4 教師との KL（seq 0〜9、コンテキスト 2,048） | 0.1359、1 位一致 86.4%（同じファイルの llama.cpp: 0.1356 / 86.6%）。UD-IQ1_S は 0.2007、16G64G は 0.083 |
| 生成、4K プロンプト、512 トークン（3 回の中央値） | 32 GB: ドラフターあり 134.2 tok/s、なし 91.3。16 GB（`STRATA_GLM_VRAM_GB=16`）: ありで 71.7、なしで 57.9 |
| プロンプト | 8K 約 1,660〜2,080 tok/s、24K 3,082 tok/s |
| `./abyss.sh --check --quick`（32 GB） | 7 / 7 合格: ドラフターあり/なしで同じ答え、日本語、コード、ツール呼び出し、8K と 24K の needle、11 回の会話 |

品質は UD-IQ1_S と 16G64G の間です。ファイルが大きいぶん VRAM に載る割合が下がり、UD-IQ1_S より遅くなります。

### unsloth のほかの UD 3.1 セット（v1.3.115、ベータ）

[unsloth の UD 3.1](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) のほかの 6 セットも動きます。
UD-IQ4_XS（33 層）と UD-Q3_K_XL（20 層）には Q5_0 の down エキスパートがあり、v1.3.115 より前のプロンプト経路は
それを 1 つずつ FP16 に戻して読んでいました（正しいが遅い）。v1.3.115 からは、ほかの型と同じくプロンプトの MMQ を
通ります（UD-IQ4_XS の 8K プロンプト: 1,365〜1,564 → 1,769〜1,776 tok/s）。パックは `tools/iq_pack.py --fast` で
作ります。RTX 5090（PCIe Gen4 x4）+ RAM 94 GB（RAM 層 50 GB）、iGPU レーンなし。KL は NVFP4 教師との比較（seq 0〜9、
コンテキスト 2,048）、生成は 4K のプロンプトで 512 トークン、3 回の中央値:

| セット | 大きさ | KL（同じファイルの llama.cpp） | 1 位一致 | 32 GB ドラフターあり / なし | 16 GB あり / なし | `--check --quick` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| UD-IQ4_XS | 93.7 GB | 0.0767（0.0757） | 89.6% | 117.0 / 81.2 tok/s | 80.9 / 64.0 | 7 / 7 |
| UD-Q3_K_XL | 89.5 GB | 0.0840（0.0833） | 89.1% | 126.1 / 80.1 | 85.2 / 64.6 | 7 / 7 |
| UD-IQ3_XXS | 83.6 GB | 0.0988（0.0984） | 88.1% | 133.0 / 92.7 | 87.4 / 65.3 | 7 / 7 |
| UD-Q2_K_XL | 79.6 GB | 0.1200（0.1198） | 87.2% | 126.9 / 88.2 | 73.5 / 57.7 | 7 / 7 |
| UD-Q2_0 | 70.5 GB | 0.1705（0.1714） | 84.6% | 139.2 / 95.7 | 64.4 / 53.0 | 7 / 7 |
| UD-IQ1_M | 68.0 GB | 0.1898（0.1901） | 83.7% | 151.8 / 102.0 | 77.0 / 61.5 | 7 / 7 |

おすすめ: GPU 32 GB + RAM 96 GB で品質を取るなら UD-IQ4_XS（自作の 16G64G の 0.083 より良い）、速さと品質の釣り合いなら
UD-IQ3_XXS、いちばん小さく速いのは UD-IQ1_M。UD-Q2_K_XL は、品質も速さも UD-IQ3_XXS に負けます。70 GB 前後のセット
（UD-Q2_0、UD-IQ2_XS）は RAM 64 GB の機械の候補ですが、その機械では測っていません。`tools/gguf_remote_types.py` は、
ダウンロードせずにヘッダーだけからセットのエキスパートの型を読みます。`tools/test_unsloth_types.py` は、これらのセットの
型と組み合わせを、エンジンがすべて扱えることを確かめます。

### いまの選び方

上の RTX 5090 での計測（v1.3.112〜v1.3.115）から:

| GPU + RAM | ファイル |
| --- | --- |
| 32 GB + 96 GB、品質重視 | unsloth UD 3.1 の UD-IQ4_XS（KL 0.077） |
| 32 GB + 96 GB、速さと品質の釣り合い | unsloth UD 3.1 の UD-IQ3_XXS（KL 0.099、ドラフターあり 133 tok/s） |
| 16 GB + 64 GB | zephel01 の [16G64G 版](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF)（KL 0.083。16 GB でドラフターあり 96.4 tok/s） |
| 12 GB + 32 GB | zephel01 の [12G32G 版](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF)（L448: 512 個中 448 個のエキスパート、67.5 GB、ベータ） |
| いちばん小さく速い | unsloth UD 3.1 の UD-IQ1_M（KL 0.190、32 GB でドラフターあり 152 tok/s） |

## 制限

- GPU 2 枚（ベータ、v1.3.109〜v1.3.110）: `--gpus 0,1` で層を分けます。ドラフターは最後のカードに置きます。分割中は会話の
  再利用がオフです。16 GB + 16 GB の組では、`--layer-split auto` はまだ 1 枚を選びますが、層 32 で分けると長いリクエストで
  22% 速いと測られています（#35）。コストモデルを合わせ直すまでは、手で分けてください（`--layer-split 32`）。
- `--abl-build` / `--uncensor` は v1.3.84 からこのモデルでも動きます（[UNCENSOR](../UNCENSOR.ja.md)）。
- AMD: v1.3.86〜v1.3.88 の高速化のいくつかは HIP では既定でオフ、v1.3.112 のグラフは CUDA だけです。AMD のカードでは
  動かしていません。
