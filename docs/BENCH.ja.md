# 自分の PC で Abyss を測る（`--bench`、`--check`）

[English](BENCH.md)

Abyss は、GPU メモリに収まらないモデルを動かします。1 トークンごとに、使うエキスパートを GPU メモリ・RAM の層・SSD のどこかから持ってきて、一部は CPU（または iGPU）のレーンで計算します。そのうちどれが速度を決めるかは、PC によって違います。GPU、PCIe スロット、RAM、SSD、CPU の組み合わせ次第です。私たちが測れるのは手元の PC だけなので、このベンチマークはあなたの PC で回して issue に貼ってもらう前提で作っています。どの PC でも同じ条件で数字が出て、1 トークンの時間がどこで使われているかを示し、正常な実行と同じ計算結果になっているかも確かめます。

## 実行方法

先に Abyss を止めてください（ベンチマークは、Abyss が使っている GPU メモリを使います）。そのうえで次のどれかを実行します。

```
./abyss.sh --bench                 # 数分: 速度、時間の内訳、品質チェック
./abyss.sh --bench --deep          # + GPU の稼働率と、PCIe / レーンの分担率の振り分け（ボトルネック探し）
./abyss.sh --bench --full          # + 各質問を 3 回、32k トークンのプロンプト
./abyss.sh --bench --config abyss-ornith-r9700l.json    # 最後に使ったものと別のモデルを測る
```

Windows では `START-ABYSS.bat --bench` です（オプションは同じ）。結果は Abyss のフォルダーの `abyss-bench.txt` に書き出され、`--report` にも含まれます。どこにも送信されません。ファイルにはホスト名・ユーザー名・フォルダーのパスは入りません。中身を確認してから、[速度の報告](https://github.com/zephel01/Abyss/issues/new?template=bench.md)に貼ってください。

ダッシュボードと同じ起動方法で、エンジン単体に次のことをさせます。

| 段階 | 内容 | 目的 |
|---|---|---|
| decode | 決まった質問 3 つに 256 トークンずつ回答。greedy、思考なし（最初に 1 回慣らしの回答をします） | どの PC でも同じ処理なので、tok/s を比べられる |
| prefill | Abyss のドキュメントから 2k と 8k トークンのプロンプト（`--full` では 32k も） | プロンプトを読む速さ。GPU と RAM に収まらないモデルでは、SSD にしかないエキスパートを長さに関係なく 1 回のプロンプトで一通り読むので、2k のほうが 8k より 1 トークンあたり遅く出ることがあります（issue #15） |
| カウンター | 各回答のあとにエンジンが出す 1 行（`glm stat`） | 1 トークンの時間がどこで使われているか |
| PCIe | エンジンが動いている間に読んだ GPU のリンク速度 | 止まっている間は Gen1 まで落ちるので、そのとき読んだ値は当てにならない |
| SSD | モデルの GGUF を 2 GB、ディスクから読む（ページキャッシュを通さない） | エンジンが SSD から読むときの速さ |
| 品質 | 決まった 512 トークンを採点し、`bench/ref/<モデル>`（正常な実行の結果）と比べる | この GPU でカーネルが壊れていれば、KL が大きくなる |
| `--deep` | GPU の稼働率（`STRATA_GLM_GAP_PROF=1`）と、PCIe の分担率を 0 / 0.25 / 0.5 / 0.75 / 1 にした 5 回の回答 | ここで何が速度を動かしているか |

## 結果の読み方

1 行目は要約です。PC どうしを 1 行で比べられます（docs/MODELS.md の表は、こうした行から作っています）。

```
Abyss v1.3.55 | Ornith-1.5-122B-UD-Q4_K_XL | AMD Radeon AI PRO R9700 32 GB | PCIe Gen4 x4 (~7.9 GB/s), the GPU can do
Gen5 x16 | RAM 94 GB | SSD 4.9 GB/s | decode 56.4 tok/s | prompt 8k 960 tok/s | quality OK | bottleneck: gpu
```

### 1 トークンの時間の内訳

```
- a token: 17.8 ms (56.2 tokens/s)
- SSD waits: 0.0 ms a token (0%), 0.0 expert reads a token
- iGPU lane: busy 7.1 ms a token (40%), 246 experts a token
- GPU memory: holds 47% of the experts a token uses; 0.4 a token come over PCIe from RAM
- drafter: 76% of its drafts accepted
```

| 行 | 意味 | 値が悪いときに分かること |
|---|---|---|
| a token | 1 トークンの生成時間（ドラフトモデルを使うときは、採用されたトークン 1 つあたり） | - |
| SSD waits | SSD から読むエキスパートを、GPU が待った時間 | 1 トークンの 4 分の 1 以上なら、モデルが GPU + RAM に収まっていない |
| CPU / iGPU lane | レーンが自分の受け持つエキスパートの計算に使った時間（GPU と並行して動く） | 1 トークン全体に近いなら、レーンが一番遅い道になっている |
| GPU memory | 1 トークンで使うエキスパートのうち GPU メモリにすでにある割合と、RAM から PCIe で送ったものの数 | 1 トークンでたくさん送っているなら、PCIe がトークンを運んでいる |
| drafter | ドラフトモデルの推測のうち、本体が採用した割合 | 低いなら、ドラフトモデルは得より損が大きい（`--dspark` を切るか、別のドラフトモデルにする） |
| tail skip | 精度を少し落とす設定が入っている（その KL は docs/MODELS.md に） | - |

### 判定

カウンターから、一番ありそうなボトルネックを判定します。目安の規則で、R9700 と RTX 5090 で確かめています。

| 判定 | 規則 | 速くする方法 |
|---|---|---|
| **ssd** | SSD の待ちが 1 トークンの 25% 以上 | GPU + RAM に収まる小さい量子化、RAM の層を増やす（`STRATA_GLM_RAM_GB`）、速い SSD。答えが少し変わってよければ、SSD の tail skip（`STRATA_GLM_TAIL_SKIP=4` + `STRATA_GLM_TAIL_SKIP_DISK=1`。R9700 の DeepSeek V4.1 UD-IQ2_XXS で +34%、KL +0.007） |
| **cpu** | レーンの稼働が 1 トークンの 80% 以上 | `./abyss.sh --calibrate`（レーンの分担とスレッド数。AMD では振り分けで一番速かった分担率を `STRATA_GLM_PCIE_SHARE` に）、CPU の電源モード、Radeon APU なら iGPU レーン |
| **pcie** | 1 トークンで 10 個以上のエキスパートを PCIe で送っている | もっと良いスロット（Gen4 x16 は x4 / OCuLink の 4 倍運べる）、`--calibrate`（AMD では `STRATA_GLM_PCIE_SHARE`） |
| **gpu** | 上のどれでもない（または `--deep` でドラフトモデルなしのとき、GPU の稼働率が 85% 以上） | `./abyss.sh --gpu-tune` と報告。この GPU 用のカーネルに改善の余地があるかもしれない |

RAM の層を増やすと SSD からの読み込みは減りますが、RAM に置いたエキスパートは CPU のレーンが計算するので、SSD の待ちが減った
分の一部はレーンの時間に回ります。DDR4 のデスクトップ（123 GB、Gen5 x16、iGPU なし、DeepSeek V4.1）で `STRATA_GLM_RAM_GB` を
80 → 88 にすると、SSD の待ちは 1 トークンあたり 11.1 → 8.7 ms に減り、レーンは 9.4 → 10.6 ms に増えて、生成は 18.0 →
18.6 tokens/s（+3%）でした。SSD の行だけでなく、両方の行を前後で比べてください。

SSD で詰まっている例です（同じ PC で DeepSeek V4.1 Flash 1017 UD-IQ2_XXS、300 GB）。

```
- a token: 50.1 ms (20.0 tokens/s)
- SSD waits: 19.5 ms a token (39%), 8.5 expert reads a token
- iGPU lane: busy 6.4 ms a token (13%), 96 experts a token
- likely bottleneck: the SSD: the experts that fit neither in GPU memory nor in the RAM tier are read from disk every token
```

「gpu」は、ほかに目立つものがないときにも出ます。メモリの各層が間に合っていて、残る改善先は GPU 自身のカーネル（とトークン間のホスト側の処理）、という意味です。私たちが持っていない GPU からの報告が一番役に立つのは、ここです。

### `--deep`：分担率の振り分け

GPU メモリにないエキスパートを、エンジンは 2 つの道に振り分けます。レーン（RAM から読んで CPU / iGPU で計算）と、PCIe（GPU に送って計算）です。`--deep` は、この分担率を 5 段階に変えて、同じ質問に答えさせます。その前後に 1 回ずつ、エンジン自身が選んだ分担率でも答えさせます。振り分けはベンチの最後に回るので、RAM の層が温まっていて、回答も短くなります。そのため、上の生成速度ではなく、同じ条件で測ったこの 2 回と比べます。どの回も同じ答えを書くので、最初に時間を測らない 1 回を回して、その答えのエキスパートを先に RAM の層や GPU に入れておきます。分担率は、エンジン自身の 2 回のうち速い方を上回ったときだけ「速い」と判定します（issue #19：まだ冷えていた 1 回目は SSD を 1 トークンあたり 6.8 回読み、後の回は 0.4 回でした。2 回の平均と比べたせいで、分担率 0 がエンジン自身より 22% 速く見えましたが、温まった後はエンジン自身の方が 5% 速かった）。1 回目の SSD の読み込みが最後の回よりずっと多いときは、振り分けがそう書きます。その場合、最初の方の分担率は分担率のせいではなく、SSD のせいで遅くなっています。

```
  share 0.00:   58.4 tokens/s  <- fastest
  share 0.25:   23.3 tokens/s
  ...
  share 1.00:   12.2 tokens/s
  the engine's own choice is as fast as the best share; share 1.00 is 4.8x slower - PCIe is the narrow path here
```

- 一番速い分担率が、エンジン自身の分担率をはっきり上回る：試す候補で、まだ設定値ではありません。振り分けは、エンジン自身の分担率で GPU のエキスパートの置き場が埋まった後に回ります（PCIe で送ったエキスパートは VRAM に上がり、分担率を下げるほど上がる数が減ります）。そのため、低い分担率を起動時から使うと、エンジン自身の分担率より遅くなることがあります。DDR4 のデスクトップの Ornith では、分担率 0 が振り分けで 34.7 tokens/s、起動時から使うと 21、エンジン自身の分担率は 24.7 でした（issue #14）。config の `"env"` に `"STRATA_GLM_PCIE_SHARE": "<分担率>"` を書いて試し、普通の `--bench` が速くなったときだけ残してください。
- エンジンの既定がすでに最速で、変化が急：片方の道がずっと細いということです。この例では PCIe（OCuLink 経由の Gen4 x4）で、エキスパートを 1 つ PCIe に回すたびに、レーンで計算するより時間がかかります。
- 変化がほとんどない：PCIe もレーンも速度を決めていません。GPU（か SSD）を見てください。

### 品質チェック

```
quality: OK (KL 0.0078 vs a known-good run, same top token 98.2%)
```

エンジンが、決まった文章（英語と日本語）512 トークンを 1 トークンずつ採点します。そして各位置の上位 32 候補の対数確率を、同じモデルが私たちの PC で出した値と比べます。参照データは `bench/ref/<モデル>.json` と `.topk` で、`tools/bench_ref.py` で作っています。同じ PC でも KL は 0〜0.01 ほど出ます。どのエキスパートを GPU とレーンのどちらで計算するかが実行ごとに少し変わり、両者で丸めの誤差が違うためです。別の GPU では、その計算の誤差が少し上乗せされます。判定は、0.02 までが **OK**、0.05 までが **slightly off**、それを超えると **OFF** です。OFF のときは、この PC が正常な実行と違う計算をしています（issue #9 のような、その GPU でのカーネルの不具合）。報告してください。

**slightly off** は、GPU が小さい PC では普通に出ます。参照データは 32 GB のカードで作っていますが、16 GB のカードでは CPU レーンで計算するエキスパートがずっと多くなり、CPU は GPU と丸め方が違うからです（issue #15。こちらで GPU を 16 GB に制限すると、GLM-5.3 Maya-S-v2 は KL 0.000 → 0.025、V4.1 UD-IQ2_XXS は 0.008 → 0.012 でした）。不具合を疑うのは **OFF** のときだけです。

- 品質チェックには、同じモデルのファイルが必要です。別の量子化や、上乗せをしたパック（`tools/dense_overlay.py`）では「skipped」になります。
- わざと数値を変える設定（`STRATA_GLM_TAIL_SKIP*`、`STRATA_GLM_ABL`、`STRATA_GLM_DEFER`）は、採点のときだけ外し、結果にそう書きます。
- 参照データがあるモデル：Ornith-1.5-122B UD-Q4_K_XL、DeepSeek V4.1 Flash 1017 UD-IQ2_XXS、DeepSeek V4 Flash 0731 UD-Q2_K_XL、GLM-5.3-Flash Maya-S-v2 IQ2_XXS

### ドラフターチェック

```
drafter check (greedy, the first answer): the same first 140 of 256 tokens with and without speculative decoding (a late near-tie: fine)
```

最初の答えを、もう一度 1 トークンずつ（ドラフトも検証窓も使わずに）出させて比べます。ドラフトが決めるのは「窓で何トークンをまとめて確かめるか」だけで、次に来るトークンは変えません。なので、2 つの答えは本来一致します。ただ、窓はトークン経路と足し算の順序が違うため、僅差の候補がある箇所では、答えの後半で分かれることがあります。これは正常です。最初の 12 トークン以内で分かれたとき（**TOO EARLY**）は、ドラフターか検証窓の計算がおかしいので、報告してください。v1.3.83 では、CUDA の検証窓でエキスパートの重みが消える不具合を、このようにして見つけて直しました。

## この PC で何が速さを止めているか

`abyss-bench.txt` と `abyss-check.txt` の最後には、その回の実測値だけから作る節があります。ドラフターなしで書いた答え（ドラフターチェックの 2 本目）の 1 トークンを部分ごとに分け、それぞれを何を変えれば速くなるかを、見込める伸びが大きい順に並べます。

```
A token without the drafter: 9.1 ms (110.5 tokens/s). Where it goes (estimates; the parts overlap a little):
- GPU compute: 6.0 ms (66%)
- CPU lane: 2.8 ms (31%), 142 experts a token
- PCIe fetches: 0.0 ms (0%), 0.0 experts a token
- SSD waits: 0.0 ms (0%), 0.0 reads a token
- host gaps (the GPU idle between tokens): 0.3 ms (3%)

What would change it (largest estimated gain first; tokens/s without the drafter - with it a VRAM gain can be larger: its verify windows give the lane more work):
1. VRAM: holds 64% of the experts a token uses - estimate: with 16 GB more (16 -> 32 GB class) ~82%; speed ~96 tokens/s (+21%) - the CPU lane takes 100% of the misses; measured on one model, the real gain was a little larger than this estimate
2. VRAM: holds 64% of the experts a token uses - estimate: with 8 GB more (16 -> 24 GB class) ~74%; speed ~89 tokens/s (+11%) - the CPU lane takes 100% of the misses; measured on one model, the real gain was a little larger than this estimate
3. PCIe: the GPU's link ran at Gen4 x4 (an expert copy measured 6.7 GB/s) - an x16 Gen5 slot (~63 GB/s) would change little here: the CPU lane already does the missed experts faster than such a copy
4. RAM: the RAM tier holds every expert the GPU does not (no SSD reads while writing) - no change needed
5. SSD: read 6.2 GB/s; waits under a tenth of a token - no change needed
```

| 部分 | 測るもの | 見積もり |
|---|---|---|
| GPU の計算 | 1 トークンの残り | 見積もらない。速い GPU のクラスほど縮む |
| CPU / iGPU レーン | 1 トークンあたりの稼働時間（`glm stat`） | 下の VRAM と PCIe の行で動く |
| PCIe の転送 | 1 トークンでコピーしたエキスパートの数 × エンジンが起動時に測った 1 回のコピー時間 | 同じコピーを、GPU の世代の x16 スロットで行った場合（公称の速さと、実際に動いたリンクの比）。外れたエキスパートは、レーンとコピーの速いほうに回る |
| SSD の待ち | エンジンが SSD を待った時間 | RAM：モデルが GPU + RAM に収まれば消える（足りない量も出します）。SSD：読み出しの速さと、一般的な NVMe Gen4（約 6 GB/s）の比 |
| トークン間のすき間 | トークンの間に GPU が空いた時間（`--check`、`--bench --deep`） | 見積もらない。CPU のシングルスレッド性能と電源設定で決まる |
| VRAM | 1 トークンで使うエキスパートのうち、すでに VRAM にあった割合 | 実行中に書き出したエンジン自身の経路を、大きさごとのキャッシュに流し直す。外れ × 実測の 1 回の時間 |

VRAM の見積もりは、経路を層ごとの LRU（最近使われていないものから追い出す）キャッシュに流し直して、今の 1 層あたりのスロット数と、8 GB / 16 GB 増やした場合を比べます。エンジンの置き方は理想的なキャッシュより追いつくのが遅く、Qwen3.8 を 12 / 16 / 24 GB でシミュレーションしたときは、外れの減り方が理想的なキャッシュの比の平方根くらいでした。そのため見積もりも平方根を取ります。まずヒット率を出し、速さは外れ 1 回の実測の時間から出します。実測と突き合わせました（v1.3.112、Qwen3.8 16G64G、ドラフターなし、16 GB → 32 GB）。ヒット率は見積もり 82%、実測 82%。速さは見積もり +21%、実測 +26%（71.4 → 90.0 tok/s）で、実際の伸びのほうが少し大きくなりました。ドラフターありでは、同じモデルで 89.2 → 131.3 tok/s（+47%）と、さらに大きく伸びます。1 トークンの 1 割に満たない部分は「no change needed」（変える必要なし）と出ます。どれも見積もりです。部分どうしは少し重なっていて、モデルや作業が変われば数字も変わります。この節に出るのは大きさとクラスだけで、製品名は出しません。

## 安定性のチェック（`--check`）

`--bench` は速さを測ります。`--check` は、インストールしたモデルを普段の使い方で動かして、落ちないか、答えが正しいかを確かめます。不具合の報告で最初に見るのはこの結果で、リリースの前にも同じチェックを回しています。

```
./abyss.sh --check                 # 15〜30 分ほど
./abyss.sh --check --quick         # 5 分ほど
./abyss.sh --check --long          # + コンテキストが許す一番大きなプロンプト
./abyss.sh --check --config abyss-q38-16g64g.json   # 最後に使ったもの以外のモデル
```

先に Abyss を止めてください。Windows では `START-ABYSS.bat --check` です。セットアップの前（または `--setup`、`--gpu`、`--gpus` を付けたとき）は、これまでどおり PC だけを調べます。チェックは、モデルのサーバー（ダッシュボードと同じく、インストールした設定で `serve/server.py`）を、この PC の中だけで空いているポートに立てて、次のものを送ります。

| ステップ | 内容 | 合格の条件 |
|---|---|---|
| drafter | 貪欲法の答え（思考なし）を、ドラフターありとなしで 1 本ずつ | 同じトークン列。12 トークン目以降で分かれるのは、足し算の順序による僅差で正常。それより前で分かれたら WARN（`--bench` が 2 つのトークンの点数を比べます） |
| ja-explain, ja-arith | 日本語の質問（`--quick` は 1 問目だけ） | 日本語で、崩れていない（U+FFFD やループがない）。計算の答えが正しい |
| code-* | Python の小さな課題（`--quick` は 1 問） | 答えのコードがテストに通る（この PC の一時フォルダで、ネットワークなし、最大 20 秒） |
| tool-call | `get_weather` の呼び出し 1 回と、その結果を受けた答え（モデルのテンプレートがツールに対応しているとき） | 引数が JSON の呼び出しと、結果を使った答え |
| long-* | 8K、24K、32K トークンと、コンテキストの上限近く（64K まで。`--long` は 64K、128K、コンテキスト全体）のプロンプトに、答えを 1 つ埋め込む。ドラフターあり | 落ちない。埋め込んだ答えが見つかる |
| conversation | 1 つの話題で 11 回のやりとり。最初に伝えた 2 つの事実を思い出させ、最後は日本語でまとめさせる | 落ちない。話題からそれない。2 つとも思い出せる。ほとんどの回で会話を使い回す |

各行には、生成とプロンプトの tok/s、使い回したトークン数、VRAM のヒット率も入ります。失敗したステップは、エンジンのログとサーバーの最後の行をファイルに残します。エンジンが落ちたら、その後のステップは実行しません（NOT RUN と出ます）。終了コードが 1 になるのは、FAIL（落ちた、答えが違う、埋め込んだ答えが見つからない、コードがテストに通らない）があるときだけです。WARN は注意で、終了コードは 0 です。ファイルには、この PC のこと（GPU、VRAM、ドライバー、動いている間の PCIe のリンク、CPU、RAM、OS、SSD の読み出しの速さ）、Abyss の版、モデル（フォルダ名と量子化）、そして「この PC で何が速さを止めているか」も入ります。`abyss-check.txt` は Abyss のフォルダにでき、`--report` にも含まれます。エンジンのログは `abyss-check.log` です。どこにも送りません。ファイルにはホスト名、ユーザー名、フォルダのパスは入りません。不具合は [bug report](https://github.com/zephel01/Abyss/issues/new?template=bug.md) を開いて、`abyss-report.txt` を添えてください。

## 対応環境

| | Linux・AMD | Linux・NVIDIA | Linux ARM・DGX Spark | Windows・NVIDIA | Windows・AMD |
|---|---|---|---|---|---|
| 速度、カウンター、判定、品質チェック、`--deep` | ○ | ○ | ○（v1.3.113） | ○ | ○ |
| PCIe のリンク速度 | ○（sysfs。ルートポートまでで一番細いリンク） | ○（nvidia-smi） | –（GPU がシステムのメモリを共有するので「memory unified」と出ます） | ○（nvidia-smi） | × |
| SSD の速度 | ○ | ○ | ○ | ○（キャッシュを通さない読み出し） | ○（キャッシュを通さない読み出し） |

AMD のカードは、カード内部のスイッチを x16 と報告します。ベンチマークはルートポートまでたどるので、OCuLink の x4 やチップセット経由のスロットも、実際のとおりに出ます。

## 報告に添えると助かるもの

- `./abyss.sh --check`（上）：この PC で落ちる、答えがおかしい、を確かめます。`--report` に `abyss-check.txt` も入ります。
- `./abyss.sh --gpu-tune`（1 分ほど、モデル不要）：GPU ごとのカーネルの選び方。`abyss-gpu-tune.txt` も貼ってください。
- `./abyss.sh --report`：PC、セットアップ、エンジンのログの速度の行（`abyss-report.txt`。ベンチマークとチェックの結果も含みます）。
- 変更の前後：`--bench` を 2 回回して、両方を貼ってください。何が変わったかが、カウンターに出ます。
