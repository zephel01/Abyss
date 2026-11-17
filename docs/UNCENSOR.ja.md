# 無検閲化（`--uncensor`）

[English](UNCENSOR.md)

モデルを拒否なしで動かすことについては、すべてこのページにまとめています。
- 実行時の編集（`--uncensor`、v1.3.71）
- 自分のモデル用にファイルを作る方法
- 確かめ方
- 実測
- 以前からある方法（abliteration 自身の編集、オーバーレイの shard、abliterated の GGUF）

この編集は拒否を外すだけで、知識は増えません。出力の扱いは使う人の責任です。

## 目次

- [どの方法を使うか](#どの方法を使うか)
- [仕組み](#仕組み)
- [有効にする](#有効にする)
- [自分のモデル用にファイルを作る](#自分のモデル用にファイルを作る)
- [ファイルを確かめる](#ファイルを確かめる)
- [自分の依頼文を足す](#自分の依頼文を足す)
- [実測](#実測)
- [制限](#制限)
- [以前からある方法](#以前からある方法)
- [ファイル形式](#ファイル形式)

## どの方法を使うか

| 方法 | 大きさ | モデル | リクエストごとの入り切り | 品質の損失 |
| --- | --- | --- | :---: | --- |
| **`--uncensor <file>`、`--abl-build` で作ったファイル**（おすすめ） | 16〜82 KB | `--glm-pack` で動くモデルすべて: DeepSeek V4 / V4.1、GLM-5.3-Flash、qwen35moe（Ornith、Qwen3.5）、Qwen3.8-Flash-Next（qwen4exp） | できる | 文章の KL 0.013〜0.053 |
| `--uncensor` に abliteration 自身の編集（mrfakename の V4.1 `abliteration-edit.npz`、ABL1） | 43 KB | V4.1 | できる | UD-IQ1_S で KL +0.002 |
| オーバーレイの shard（`tools/dense_overlay.py --abl`） | 1.4 GB | V4.1 | できない | KL +0.0006（テンソルをもう一度丸める） |
| abliterated の GGUF（huihui-ai の V4 0731 Q2、orcarouter の V4.1 Mix-256G、orcarouter の Qwen3.8-Flash-Next-Uncensored） | モデル 1 本分 | そのモデル | できない | V4.1 Mix-256G: 文章の KL 0.083 |

実行時の編集の利点は 3 つあります。
- モデルの 2 本目のコピーが要りません。
- 元のモデルにはリクエスト 1 つで戻せます。
- V4.1 では、KL が abliterated の GGUF の 3 分の 1 以下です。

Qwen3.8-Flash-Next（qwen4exp、[docs/qwen38](qwen38/README.ja.md) の速い経路）でも `--abl-build` と `--uncensor` が使えます
（未リリース。それまでは、この経路では `--abl-build` が何も測れず、PLE の行も編集されませんでした）。

## 仕組み

モデルの拒否は、残差ストリームの中のほぼ 1 本、または数本の方向に沿って起きます（Arditi らの「Refusal in Language Models Is
Mediated by a Single Direction」）。abliteration は、その方向に何も書き込まないように重みを書き換えます。Abyss は同じことを
動かしながら行います。残差への書き込みのたびに、その方向の成分を引きます。

```
w  <-  w - s · Q (Qᵀ w)        Q: n_embd × rank の正規直交、s: 強さ
```

かける場所は次のとおりです。

- トークンの埋め込み
- 各層の attention の出力
- 各層の FFN の出力全体（ルーティングされたエキスパート + 共有エキスパート + CPU レーンの分）
- V4.1 の Engram の加算、Qwen3.8 の層ごとの埋め込み（PLE）の行

SSD 上のルーティングされるエキスパートには触らないので、ディスクも余分な丸めも増えません。トークン経路、プロンプト経路、投機デコードの
検証窓のすべてで効きます。入り切りは GPU 上のフラグで行うので、CUDA / HIP のグラフの中でも動き、リクエストごとに切れます。

## 有効にする

**どのファイルか。** モデルの実行設定、Abyss フォルダーにある `abyss-<name>.json` です（例: `abyss-dsv41-iq2xxs.json`）。
- setup が書き出すファイルです。
- `run-abyss-<name>.sh` がサーバーに `--config` で渡すファイルで、サーバーを自分で起動するときに
  `serve/server.py --config <file>` で指定するファイルです（場所と、いま使っているファイルの見つけ方は [SETTINGS](SETTINGS.ja.md)）。
- モデルごとに 1 つあります。`.abl` ファイルを作ったモデルのものに書いてください。

一番外側に、`"exe"` や `"args"` と並べてキーを 1 つ足します（`"args"` や `"env"` の中ではありません）。

```json
{
 "exe": "/home/you/llm/apps/Abyss/build/abyss",
 "args": ["--glm-pack", "...", "--max-context", "65536"],
 "uncensor": "/home/you/llm/apps/Abyss-data/abl/dsv41-iq2xxs.abl",
 ...
}
```

書いたらサーバーを再起動します（`../Abyss-tools/abyss.sh restart`、または止めてから `run-abyss-<name>.sh` をもう一度実行）。ファイルはエンジンの起動時に読み込みます。
起動時のエンジンログ（設定の `"log"` のファイル）に `glm fast: uncensored (<file>): ...` と出ます。

ほかに 2 つ、同じことをする書き方があります。`"args"` に `"--uncensor", "<file>"` を足すか、設定の `"env"` に
`"STRATA_GLM_ABL": "<file>"` を書きます。3 つのうち 1 つだけを使ってください。

以後のリクエストは編集つきで動きます。入り切りの方法は次のとおりです。

| 入り切り | 編集 |
| --- | --- |
| リクエストに `"uncensor": false` | そのリクエストだけオフ |
| リクエストに `"uncensor": true`、またはキーなし | オン |
| `POST /settings` に `"uncensor": false/true` | 他のアプリが受け取る既定値（`temperature` などと同じ共有設定） |
| エンジンの通信: `GEN ... abl=0` / `abl=1` | オフ / オン |
| `STRATA_GLM_ABL_ON=0` | 読み込むが、既定はオフ |

もう一方の設定で計算した会話は再利用しません（サーバーは、この入り切りも含めて会話のキャッシュを見分けます）。

**オフなら何も変わりません。** ファイルを読み込んでオフにした出力は、ファイルなしとビット単位で一致しました（V4.1、score 経路で
KL 0、CPU レーンの件数も同じ）。**オンでも遅くなりません。** 差は回すたびの揺れの範囲です（V4.1、AI PRO R9700: オン 20.4 tok/s、
オフ 21.3、ファイルなし 22.2）。

## 自分のモデル用にファイルを作る

ファイルは、使うモデル自身から作ります。AI PRO R9700 で 1 時間以内です。
- モデルごとに 1 本作ってください。V4 のファイルは V4.1 には合いません。モデルの幅が違うファイルは、起動時に読み込みを断ります。
- 下の実測のファイルは、計測したのと同じ量子化で作りました。同じモデルの別の量子化に使う場合は、KL を確かめるか、
  その量子化用に作り直してください。

**1. 質問を用意する。** 方向は、公開されている質問のセットで測ります。
- AdvBench
- MaliciousInstruct
- ForbiddenQuestions
- SimpleSafetyTests
- JailbreakBench（有害な行動と無害な行動）
- XSTest（危険な質問と安全な質問）

それぞれの配布元から `Abyss-data/datasets/safety` に取ってきてください（ファイルの一覧は `MANIFEST.txt` にあります）。
`tools/abl_prompts.py` が、これらを均等に混ぜ、pack のチャットテンプレートで（thinking はオフで）トークン ID にして書き出します。

```sh
.venv/bin/python -I tools/abl_prompts.py --pack <pack> --out prompts.txt
```

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--train N` | 64 | 測定に使う有害・無害の質問の数（それぞれ） |
| `--check N` | 24 | その次の質問を、候補を試すのに使う数 |
| `--exclude FILE ...` | – | 評価に使う質問を除く（1 行 1 問のテキスト、または `"prompt"` を持つ JSON lines） |
| `--harmful mix\|advbench` | mix | AdvBench だけだと、GLM に弱い拒否（海賊版、医療、詐欺）が残る |
| `--cyber N` | 0 | マルウェアや不正アクセスの質問を N 問、有害側に必ず入れる（ForbiddenQuestions の Malware カテゴリと、AdvBench / MaliciousInstruct の cyber 系の言葉。JailbreakBench からは取らない）。編集のあとでもマルウェアの拒否が残るときに使う。Qwen3.8 では、この拒否が別の方向だった |
| `--data DIR` | `Abyss-data/datasets/safety` | セットの置き場所 |

どの質問を選ぶかは毎回同じです。データセットは信頼できないデータとして扱い、この道具は `python -I` で動かしてください。

**2. 作る。**

```sh
build/abyss --glm-pack <pack> --max-context 4096 --abl-build model.abl --abl-prompts prompts.txt \
  --abl-gen 32 --abl-rank 3 --abl-range 0.3,0.9 --abl-candidates 9
```

中でやっていること:

1. 測定用の質問を最後のトークンまで読み、各層の残差を記録します。
2. 層の方向 = 有害な質問の平均 − 無害な質問の平均。これを**無害な質問の平均たちと直交させます**。この手順がないと、モデルが
   常に持つ巨大な固定チャネルが方向に混ざり、モデルが壊れます（あるモデルで KL 9）。
3. 候補の層ごとに、確認用の有害 24 問・無害 24 問で方向を試します。見るのは、有害な質問の答えがいくつまだ断るかと、
   無害な質問の 1 トークン目の KL です。
4. `--abl-max-kl` 以内の良い層を `--abl-rank` 本まで組み合わせ、いちばん良いもの（ABL2）を書き出します。全層の方向は
   `model.abl.dirs` に残します。

| オプション | 既定 | 意味 |
| --- | --- | --- |
| `--abl-candidates N` | 8 | 試す層の数（`--abl-range` の中に均等に並べる） |
| `--abl-range LO,HI` | 0.2,0.8 | 候補の深さ（全層に対する割合） |
| `--abl-layer L` | – | この層だけ試す |
| `--abl-gen N` | 0 | 有害な質問の答えを greedy で N トークン生成し、拒否の言葉（" can't"、" cannot"、"Sorry"、申し訳 など）を探す。32 を推奨。GLM は見出しから書き始めて 2 行目で断るので、1 トークン目だけでは見逃す |
| `--abl-max-kl K` | 0.2 | 候補に許す、無害な質問の 1 トークン目の KL（大きめに。下を参照） |
| `--abl-rank N` | 1 | 上位 N 本の方向を組み合わせたものも試す（1〜4） |
| `--abl-scale S` | 1 | 強さ。1 で成分をちょうど取り除き、1 より上は引きすぎる |
| `--abl-dirs F` | – | 前回の `.dirs` を使い回す。測定を飛ばすので、別の設定を 10 分ほどで試せる |

ログは候補ごとに 1 行です（V4 0731、抜粋）:

```
ABL unedited: a refusal word in the first 32 tokens of 20 of 24 harmful check answers
ABL layer 26: ... harmless KL 0.7945, refused in 32 tokens 7
ABL layer 29: ... harmless KL 0.7295, refused in 32 tokens 3
ABL layer 39: ... harmless KL 0.3053, refused in 32 tokens 6
ABL chose layer 29 (rank 1, scale 1.30: ...) -> model.abl
```

チャット形式の質問で測る 1 トークン目の KL は、品質の損失を大きく見せます。V4 では 0.7〜1.0 と出ましたが、普通の文章の KL は
0.013 でした。拒否の数で選び、そのあと文章の KL を確かめてください（次の節）。V4 では `--abl-max-kl 1.5` にして、いちばん良い層を
選ばせました。

## ファイルを確かめる

**品質**: score 経路で、普通の文章の KL を測ります。ファイルなしで 1 回、ファイルありで 1 回です。

```sh
STRATA_GLM_SCORE=400 STRATA_GLM_SCORE_DUMP=ref.bin        # ファイルなし: 基準を作る
STRATA_GLM_SCORE_REF=ref.bin STRATA_GLM_ABL=model.abl     # ファイルあり: "vs ref: KL ..." が出る
```

（ファイルを読み込んで `STRATA_GLM_ABL_ON=0` にしたときは、KL 0 になるはずです。）

**拒否**: `prompts.txt` に入っていない質問を、`"uncensor": false` と `true` で、greedy で聞きます。答えの書き出しを読んで
判定してください。無害な質問の数も数えてください。良い編集は、断りすぎ（無害な質問を断ること）も減らします。

## 自分の依頼文を足す

**使うとき。** `--abl-build` のあと、ほかは答えるのに、ある種類の依頼だけ拒否が残るとき。拒否は種類ごとに別の方向に
乗っていることがあり、作成は、依頼文に出てくる方向しか見つけません。Qwen3.8-Flash-Next（ISTA IQ3_S）では、最初の
ファイルで Malware / Hacking の拒否が残りました（JailbreakBench の 5 問中 4 問）。cyber 系の依頼文を足し、方向を 1 本
増やして作り直すと消えました。

| 作成 | JBB の拒否 | Malware / Hacking | XSTest の過剰拒否 | 数学 | 文章の KL（en / ja / code / md） |
| --- | :---: | :---: | :---: | :---: | --- |
| 標準の mix、`--abl-rank 3` | 11.1% | 5 問中 4 問 | 0.8% | 44 / 50 | 0.028 / 0.015 / 0.046 / 0.029 |
| `--cyber 30`、`--abl-rank 4` | 1.2% | 5 問中 0 問 | 0.0%（50 問） | 43 / 50 | 0.027 / 0.015 / 0.057 / 0.033 |

（JailbreakBench のテスト 81 問、数学 50 問を 512 トークンまで、greedy。拒否は答え全体で判定。`--cyber 30` だけで
rank 3 のままだと、マルウェアの拒否は残りました。消したのは 4 本目の方向、18 層です。）

**`--cyber N`** は、マルウェアや不正アクセスの依頼文を N 問、有害側に必ず入れます。ForbiddenQuestions の Malware
カテゴリと、AdvBench / MaliciousInstruct の cyber 系の言葉から取ります（139 問。JailbreakBench からは取りません）。

**`--extra-harmful FILE`** は、自分の依頼文を同じように有害側に必ず入れます。Abyss は中身を配っていません。モデルが
まだ拒否する種類の依頼を、自分で書いてください。ファイルの形:

```
# 1 行に 1 つ、UTF-8。空行と # で始まる行は読み飛ばします
example prompt 1
example prompt 2
{"prompt": "example prompt 3"}
```

足した依頼文は先頭に入り、`--train` + `--check` に切り詰めても残ります（残りは標準の mix で埋めます）。
`--extra-n N` で多くても N 問だけ使います。`--exclude` はそのまま効きます。使った数はツールが表示します。そのあと、
方向を 1 本増やして、いつもどおり作ります:

```sh
.venv/bin/python -I tools/abl_prompts.py --pack <pack> --out prompts.txt --cyber 30 --extra-harmful my.txt
build/abyss --glm-pack <pack> --max-context 4096 --abl-build model.abl --abl-prompts prompts.txt \
  --abl-gen 32 --abl-rank 4 --abl-range 0.3,0.9 --abl-candidates 9 --abl-max-kl 1.0
```

**確かめ方**は[ファイルを確かめる](#ファイルを確かめる)と同じです。作成に使っていない依頼文で拒否を数え、長い答え
（数学）と文章の KL が悪くなっていないかを見ます。足すファイルは、テストに使う依頼文と分けてください。作成に使った
依頼文では、編集の効き目はわかりません。

## 実測

AI PRO R9700 で測りました。
- 質問は、作成に使っていない 101 問（有害 66、無害 35）です。
- 生成は greedy です。
- 「拒否」は答えの書き出しで判定しました。

| モデル | 有害で拒否（オフ → オン） | 無害で拒否（オフ → オン） | 文章の KL | 編集 | ファイル |
| --- | :---: | :---: | :---: | --- | --- |
| DeepSeek V4.1 Flash 1017、unsloth UD-IQ2_XXS | 86% → 3% | 17% → 3% | 0.023 | 27+30+33+24 層、強さ 1.3 | 82 KB |
| DeepSeek V4 Flash 0731、UD-Q2_K_XL attn-q5k | 85% → 6% | 11% → 3% | 0.013 | 29 層、強さ 1.3 | 16 KB |
| GLM-5.3-Flash、Maya-S-v2-IQ2_XXS | 88% → 8% | 11% → 3% | 0.053 | 31+28+34 層、強さ 1.0 | 50 KB |

Qwen3.8-Flash-Next、ISTA-DASLab GSQ-RCO IQ3_S（RTX 5090）: 作成は 39+43+36 層、rank 3、強さ 1.0 を選びました（31 KB、
`--abl-gen 32 --abl-range 0.3,0.9 --abl-candidates 9 --abl-max-kl 1.0`、5 分）。有害な確認用の 24 問のうち、最初の 32 トークンに
拒否の言葉が出た答えは、編集なしで 23 問、ありで 3 問です。

有害な質問への答えの内訳（オン）:

| モデル | 拒否 | 短い注意書きのあとに答える | そのまま答える |
| --- | :---: | :---: | :---: |
| V4.1 | 3.0% | 22.7% | 74.2% |
| V4 | 6.1% | 28.8% | 65.2% |

比較: orcarouter の abliterated V4.1 Mix-256G（重みを作り直したもの）は、文章の KL が 0.083 です。

## 制限

- **拒否は少し残ります。** 残るものの多くは、短い注意書きのあとに答えるもの、医療や投資の助言です。
- **ファイルを読み込んでいる間、速度の機能が 2 つ止まります。** CPU レーンの答えの hc3 融合と `STRATA_GLM_DEFER` です（FFN の
  出力がそろってから編集するため）。ファイルをオフにしていても止まったままです。
- **ファイルはモデルごとに 1 本です。** 同じモデルの別の量子化では、これらのファイルを測っていません。KL を確かめてください。
- **`--glm-pack` のモデルだけです。** 古いエンジン経路（Maya 時代の pack、Qwen3.8-Flash-Next の古い pack）では使えません。

## 以前からある方法

**abliteration 自身の編集**（v1.3.53）。mrfakename/DeepSeek-V4.1-Flash-1017-abliterated は `abliteration-edit.npz`（43 KB）を
配っています。中身は rank 2 の拒否の部分空間と、attention と共有エキスパートの出力にかける層ごとの強さです。変換すると、
`--abl-build` のファイルと同じように読み込めます。

```sh
.venv/bin/python -I tools/abl_export.py abliteration-edit.npz v41-mrfakename.abl
```

UD-IQ1_S での結果:
- KL: 0.3557 → 0.3577
- 拒否を誘う 16 問（英語 12、日本語 4）: 断ったのは 12 問 → 0 問

編集するのはこの 2 つの出力だけです。101 問のセットでは測っていません。

**オーバーレイの shard**（v1.3.52）。同じ編集を重みに書き込む方法です。`attn_output_b` / `ffn_down_shexp` に
`W - s Q Qᵀ W` をかけ、62 テンソル・1.4 GB のオーバーレイ GGUF を作ります。

```sh
.venv/bin/python -I tools/dense_overlay.py <モデルの shard 1> <出力先> --abl=<dir>/abliteration-edit.npz
.venv/bin/python tools/iq_pack.py --gguf <出力先>/overlay-00001-of-0000N.gguf --out <出力先>/pack --compat-bf16 --first-wins
cp <モデルのディレクトリ>/pack/expert_*.txt <出力先>/pack/
```

UD-IQ1_S での結果:
- KL: 0.3557 → 0.3563
- 拒否: 16 問中 14 問 → 0 問
- Q5_K のオーバーレイと一緒に使うと KL 0.3738（2 回丸めるため）

リクエストごとには切れません。実行時の編集がこれに代わります。

**abliterated の GGUF** は、そのモデルの他の GGUF と同じように pack して動かします（[MODELS](MODELS.ja.md)）。
- huihui-ai の DeepSeek V4 0731 Q2（`--compat-bf16`）
- orcarouter の V4.1 Mix-256G（文章の KL 0.083、swe-bench 22.4 tok/s、ディスク律速）
- orcarouter の Qwen3.8-Flash-Next-Uncensored

## ファイル形式

リトルエンディアンです。

```
char[4]  "ABL1" | "ABL2"
int32    n_embd, rank, layers
float32  Q[n_embd][rank]          行優先、列が正規直交
float32  strength[layers][2]      層ごと: ABL1 は attention と共有エキスパート、ABL2 は attention と FFN 全体
float32  埋め込みの強さ            ABL2 のみ
```

ABL1 は attention と共有エキスパートの出力を編集します（abliteration 自身の編集）。ABL2 は残差への書き込みすべてを編集します。
`n_embd` がモデルと違うファイルは、起動時に読み込みを断ります。
