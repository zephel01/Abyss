# Qwen3.8-Flash-Next を Abyss で動かす

English: [README.md](README.md)

Qwen3.8-Flash-Next（llama.cpp の `qwen4exp`）は、v1.3.74 から Abyss の速い経路（`--glm-pack`）で動きます。プロンプト
経路、検証窓、MTP ドラフターも使えます。古いエンジン経路（[docs/maya/ORCA.md](../maya/ORCA.md)）もそのまま動きますが、
速い経路のほうがプロンプトは数倍、答えも速くなります。

計測は orcarouter の
[Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) IQ3_XXS
（2 シャード、85.2 GB）で、AI PRO R9700（32 GB、PCIe Gen4 x4）、RAM 94 GB（`STRATA_GLM_RAM_GB=60`）、iGPU レーンありの
環境です。NVIDIA ではこの経路をまだビルドも計測もしていません。

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
0.035 / 90.9% でした。このモデルでは llama.cpp 自身の HIP と CPU の差が KL 0.045 です。貪欲生成では、下書きがあっても
無くても同じトークンが出ます。

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

選び方: 上から順に見て、常駐分が「GPU のメモリ + エンジンが使える RAM（RAM のおよそ 75%、`STRATA_GLM_RAM_GB`）」に
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

## 制限

- GPU は 1 枚: 複数枚への分割はこのモデルでは試していません。ドラフターは 1 枚でモデル全体を動かすときだけ読み込まれます。
- `--uncensor` の編集ファイルはこのモデル用にはありません（ここで使った GGUF はもともと無検閲版です）。
- この経路の NVIDIA でのビルドは確認していません。
