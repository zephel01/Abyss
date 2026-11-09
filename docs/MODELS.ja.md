# モデル評価と GPU 別の参考ベンチマーク

どのモデルファイルを選ぶか、手元の PC でどのくらいの速さが出るかの目安です。数値はすべて Abyss で測ったもので、
行ごとに Abyss のバージョンか日付を書いています。速さは tok/s（ms/token からの換算は 1000 ÷ ms）。
「未計測」は測っていない、「–」は該当なしです。答えが変わる（lossy な）設定には、その KL を横に書いています。

English: [MODELS.md](MODELS.md)

## 測り方

| 項目 | 内容 |
| --- | --- |
| V4.1 の KL / top-1 / ppl | 量子化前の DeepSeek-V4.1-Flash-1017 の top-256 log-prob に対する KL。Lucebox の held-out 8 系列（8 × 1,023 位置、[Lucebox/...-ROCMFP-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) の `eval/`）。小さいほど元のモデルに近い |
| V4 0731 の KL | 同じファイルを llama.cpp で動かした出力に対する KL（3,000 トークン）、またはエンジン自身の参照出力に対する KL（1,000〜3,000 トークン）。量子化そのものの損失ではなく、エンジンの正しさと設定の影響を見る値 |
| GLM の nll | `glm_ids4k` の 3,000 トークンを流し、後半 1,000 トークンの nll（小さいほど良い）。KL（`ref4k_glm` 基準）は別の量子化の出力との差で、品質の値ではない |
| Ornith の KL | 同じ GGUF の llama.cpp（CPU バックエンド）に対する KL。llama.cpp 自身の CPU と HIP の差が 0.027〜0.029 |
| swe-bench | 自前のコーディング問題（llmbench）。**60 問 × 5 回**（解決率と pass@1）、**L7 = 難問 16 問**（× 3 または × 1）、速さを見る **6 問**（t003 / t010 / t018 / t025 / t030 / t036、サンプリングあり、±5% 揺れる）、V4.1 用の **8 問**（2026-10-23〜、greedy）。tok/s はサーバー経由の通しの値 |
| 1 トークンずつ | ドラフトなし（または n-gram のみ）の生成の速さ。swe-bench より遅く出る |

試験機: Ryzen AI Max+ 395（Radeon 8060S、RAM 94 GB）に、OCuLink（PCIe Gen4 x4）で RTX 5090 32 GB か AI PRO R9700 32 GB を
つないだもの。モデルの SSD は 2026-10-23 夜に ORICO（約 3.4 GB/s）から Lexar NM790（約 6.7 GB/s）に替えた。V4.1 は SSD の速さで
数値が変わるので、どちらの SSD かを書いています。

## 1. モデル評価

### DeepSeek V4 Flash 0731

人気（Hugging Face の過去 30 日、2026-10-24 時点）: huihui-ai の 0731 abliterated Q2（月 32 万）、huihui-ai の 0422 ds4（23 万）、
unsloth の 0731（12 万）、0xSero の EXL3（11 万）。この大きさ（約 87〜100 GB）は 32 GB の GPU + 約 92 GB の RAM に収まり、SSD を読まない。

| ファイル（入手先） | サイズ | 品質 | 速さ | 必要なメモリの目安 | おすすめ |
| --- | ---: | --- | --- | --- | --- |
| `DeepSeek-V4-Flash-Q2-0731.gguf`（[huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF)、`--compat-bf16` で pack） | 86.8 GB | llama.cpp との KL 0.0075、ppl 6.32（v1.2.2）。swe-bench 解決率は未計測 | 5090: swe 6 問 91.1 tok/s（v1.3.27）。R9700 + iGPU レーン: swe 6 問 38.1（v1.3.39） | VRAM 32 GB + RAM 約 70 GB | 制限解除版が要るとき。セットアップは手作業 |
| huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF（0422） | 約 87〜100 GB | 未計測 | 未計測 | – | 動くか未確認 |
| `UD-Q2_K_XL`（[unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)、セットアップの既定。attention を Q5_K にした pack が既定、KL 0.0055 → 0.0119） | 96.8 GB | llama.cpp との KL 0.0096。swe-bench 60 問 × 5: 59/60 解決、pass@1 95.8%（v1.3.27） | 5090: swe 60 問 106.7 tok/s（v1.3.27）、1 トークンずつ 92〜95。R9700 + レーン + DSpark: swe 60 問 70.2（v1.3.43 直前のビルド） | VRAM 32 GB + RAM 約 70 GB で全エキスパートが載る | **既定。まずこれ** |
| 0xSero の DGX-Spark 向け EXL3 | – | 未計測 | – | – | DeepSeek の EXL3 はまだ読めない |
| DSpark ドラフトモデル `DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf`（[bleysg](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)、`--dspark`） | 6.94 GB | 答えは変わらない（受理するのは本体のサンプルと一致したものだけ） | R9700、swe 6 問: レーンなし 48.4 → 61.9、レーンあり 51.7 → 67.3 tok/s（v1.3.43） | VRAM 6.5 GB 追加（その分のエキスパートは SSD へ） | R9700 で速さが欲しいとき。DGX Spark では 21.0 → 22.8 tok/s（v1.3.114） |

lossy な設定（V4 0731、5090、v1.2.1 / v1.3.2x）: `STRATA_GLM_TAIL_SKIP=4` は 1 トークンずつ +7%（iGPU レーン）、KL 0.0226。
`STRATA_GLM_DEFER=1` は +12〜15%、KL 0.08（オフ 0.012）、L7 の解決 14 → 13 問。`TAIL_SKIP=1` / `=2` の KL は未計測。

### DeepSeek V4.1 Flash 1017

人気（同じく 2026-10-24 時点）: V4.1 は 9 月版のほうがまだ多い（月 148 万、1017 は最初の 1 週間で 5.8 万）。Abyss が動かすのは 1017 です。
antirez/deepseek-v4.1-flash-gguf（134 万）は 9 月版と 1017 のファイルが混在。1017 だけの量子化では nazeshinjite が 1 位。

V4.1 はエキスパートが RAM に収まらず、一部を毎トークン SSD から読む。小さいファイルほど SSD の読みが減って速い。
下の「1 トークンずつ」は 8 系列の平均。

| ファイル（入手先） | サイズ | 品質: KL / top-1 / ppl（8 系列） | 速さ | おすすめ |
| --- | ---: | --- | --- | --- |
| `Q2`（[nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF](https://huggingface.co/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF)） | 366 GB | 0.229 / 79.2% / 9.69（ルーティングどおり）、`TAIL_SKIP` ありで 0.241 / 78.9% / 9.75。swe-bench 60 問 × 5: 60/60 解決、pass@1 98.4%（v1.3.33） | 5090: swe 60 問 50.6 tok/s（v1.3.33、TAIL_SKIP あり、ORICO）、1 トークンずつ 27.9（あり）/ 17.7（なし）（v1.3.31）。R9700 + レーン: 19.7（あり、v1.3.46、ORICO） | 品質の実績が最も多い。速さは UD-IQ2_XXS と同じで SSD を 66 GB 多く使う |
| antirez/deepseek-v4.1-flash-gguf の 1017 ファイル | – | 未計測 | 未計測 | 動くか未確認（9 月版は対象外） |
| `UD-IQ2_XXS`（[unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)） | 300 GB | 0.186（なし）/ 0.209（TAIL_SKIP あり）。top-1 / ppl / swe-bench は未計測 | R9700 + レーン、1 トークンずつ: 17.3（なし）/ 24.3（あり）（v1.3.46、ORICO） | **品質重視ならこれ**（v1.3.46 からの推奨） |
| `UD-IQ1_M`（unsloth） | 266 GB | 0.270（なし）/ 0.281（あり） | R9700 + レーン: 20.5 / 26.5（v1.3.46、ORICO） | 中間 |
| `LDLQ-132G` / `LDLQ-139G`（[zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-IQ2_XXS-32G96G-GGUF)、BETA） | 275.8 / 283.0 GB | 0.194 / 0.182（ルーティングどおり、8 系列。UD-IQ1_M は 0.198）。難しめの swe 10 問は両方 10/10、品質 77.6 / 81.4 | 5090 + レーン、32 GB + 96 GB（RAM 枠 60 GB）+ DSpark: swe 3 問 67.5 / 57.9 tok/s（UD-IQ1_M は 71.8）、難しめ 10 問 58.0 / 47.8（2026-11-02） | GPU 32 GB + RAM 96 GB 向けの自作 LDLQ 量子化。`132G` は速さ重視、`139G` は品質重視（[詳細](deepseek41/README.md)） |
| `Compact-JA256`（[zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-E256-IQ2_XXS-32G96G-GGUF)、BETA） | 207.1 GB（エキスパート 88.4 + Engram IQ4_NL 110.6） | 0.383 / 75.2%（ルーティング厳密、8 本）。難しめ swe 10 問 8/10、品質 80.5 | 5090 + レーン、32 GB + 96 GB（RAM 段 60 GB）+ DSpark: swe 3 問 **87.0 tok/s**、難しめ 10 問 75.7、32K の入力処理 1,833 tok/s（2026-11-03） | 384 個中 256 個を日本語・中国語・英語・コード・数学・ツールの文章で選んだ汎用版。日本語は崩れない。本格的なコーディングは `LDLQ-132G`（[詳細](deepseek41/README.md)） |
| `GSQ-RCO-Coder`（ISTA-DASLab、384 個中 192 個） | 182.7 GB | 0.582 / 69.3%（2026-11-03）。難しめ 10 問 9/10 | 5090 + レーン: swe 3 問 97.0 tok/s | コードだけで選んだ版。日本語の質問に中国語で答える |
| `UD-IQ1_S`（unsloth） | 233 GB | 0.355 / 75.0% / 10.80（なし）、0.359（あり） | 5090: 45.6（あり、v1.3.38、ORICO）。R9700 + レーン: 27.2 / 31.9（v1.3.46、ORICO）、swe 8 問 43.7、+ DSpark 54.4（2026-10-24、NM790、下の注） | **速さ重視ならこれ**。品質ははっきり下がる |
| `GSQ-RCO-IQ2_XXS`（[ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)、Engram は別ファイル） | 248.8 GB（本体 138.2 + Engram 110.6） | 0.3137（なし）/ 0.3266（TAIL_SKIP あり） | R9700 + レーン、swe 8 問: 30.7 / 30.9（なし）、42.2 / 41.3（あり）、48.3 / 47.3（あり + DSpark）。8K prompt 109 秒（約 75 tok/s）（2026-10-24、NM790） | 小さい本体で UD-IQ1_S より KL が低い。40 tok/s には TAIL_SKIP が要る |
| `GSQ-RCO-IQ1_M`（ISTA-DASLab） | 231.3 GB | 0.3845（2026-10-24） | 未計測 | KL が UD-IQ1_S より高いので勧めない |
| `Q2_K`（[mrfakename/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF)） | 247 GB | 教師 KL 0.64（dense と Engram も Q2_K） | 5090: 約 9.7 tok/s（2026-10-18） | 単体では勧めない |
| ハイブリッド: Lucebox の Q8_0 dense + 8 bit Engram と mrfakename のエキスパート | 201 GB | 0.168 / 83.4% / 9.25 | 5090: 8.5 tok/s（2026-10-18） | 最も正確だが遅い |

V4.1 のオーバーレイ（UD-IQ1_S、R9700 + レーン、v1.3.52）:

| 方法 | 追加するもの | KL（基準 0.3557） | 速さ | 用途 |
| --- | --- | --- | --- | --- |
| Q5_K dense オーバーレイ（`tools/dense_overlay.py`、opt-in、lossy） | 3.2 GB | 0.3652、ppl 10.79 → 10.93 | swe 8 問 40.1 → 43.0 tok/s（n-gram 窓） | 少しでも速く |
| Q5_K と拒否のオーバーレイを一緒に | – | 0.3738 | 未計測 | – |

拒否の編集（`--abl` のオーバーレイ、`STRATA_GLM_ABL`）と実行時の `--uncensor` の計測は [UNCENSOR.ja.md](UNCENSOR.ja.md) にあります。

V4.1 の lossy な設定（KL は nazeshinjite Q2、8 系列）: `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1` は 0.229 → 0.240〜0.241
（5090 で 17.7 → 約 28〜29 tok/s）。`=4` は 0.235。`STRATA_GLM_ROUTE_AFFINITY_RAM=0.08` は 0.2406 → 0.2438（-8〜12% / トークン）。
`STRATA_GLM_DEFER=1` は 1 系列で 0.15 → 0.28（不採用）。
R9700 + レーンの UD-IQ2_XXS（NM790、4 系列、2026-10-27）: 省略なしで 1 トークン 48.5〜53.4 ms（約 19 tok/s。うち約 20 ms が SSD 待ち）、`TAIL_SKIP=4` + `_DISK=1` で 39.8 ms（25.1 tok/s）、KL +0.007。`=3` は 38.6 ms、+0.020 で、ここでは `=4` を選ぶ。

V4.1 の DSpark ドラフトモデル（Lucebox の `deepseek41-dflash-draft`、DSpark-draft-Q2K-Q4K、4.92 GB、VRAM 約 1.67 GB、opt-in、v1.3.53）:
答えは変わらない。NM790 の R9700 で swe 8 問 +16〜19%（UD-IQ1_S: 40.0 → 47.5、2026-10-24）。ORICO クラスの SSD では効果なし（2026-10-23）。

### GLM-5.3-Flash

ダウンロード数は未調査。どれも VRAM 32 GB + RAM 約 60〜78 GB で全エキスパートが載る。

| ファイル（入手先） | サイズ | 品質: nll（小さいほど良い） | 速さ | おすすめ |
| --- | ---: | --- | --- | --- |
| Maya-S-v2-IQ2_XXS（Project Maya） | 96.5 GB | 1.634、top-1 62.3%。swe-bench L7: 13/16 解決（Abyss 1.1.2、2026-10-11） | 5090: 1 トークンずつ 約 67（14.8 ms、2026-10-19）、swe 6 問 64.1（v1.3.27）。R9700 + レーン: 34.3〜38.3（v1.3.40）、swe 6 問 35.8（v1.3.39） | **既定** |
| Maya-S24 | 94.7 GB | 未計測（本ページの尺度では） | 5090: swe 6 問 70.7（v1.3.27）。R9700 + レーン: swe 6 問 41.2（v1.3.39） | VRAM 24 GB 以下のカード向け |
| EXL3 `sc_2.10bpw`（[turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/sc_2.10bpw)、試験的） | 88.3 GB | 1.591〜1.598、top-1 63.3〜63.7%。KL 0.168（Maya-S-v2 との差） | 5090: 約 53 tok/s（18.8 ms、v1.3.36） | 小さくて文章の予測が良い。速さは約 8 割 |
| EXL3 `3.05bpw`（turboderp） | 125.3 GB | 未計測 | 未計測 | インストーラに載っているが未計測 |
| exlr8 K2（kindlingai、`tools/exlr8_convert.py --width 2` → `tools/exl3_pack.py`、v1.3.53） | 86 GB | nll 1.608〜1.613（KL 0.208 は `ref4k_glm` 基準） | R9700 + 8060S レーン、RAM 層 60 GB: 約 22.9 tok/s（43.7 ms、2026-10-24） | いちばん小さい。速さは今後 |
| exlr8 K3（kindlingai、`--width 3`、v1.3.53） | 125 GB | nll 1.548（KL 0.212 は `ref4k_glm` 基準） | R9700 + 8060S レーン、RAM 層 60 GB: 約 11.2 tok/s（89.1 ms、エキスパートの 3 分の 1 を SSD から、2026-10-24） | 測った GLM で品質が最も高い。94 GB の PC では遅い |
| exlr8 K4（kindlingai） | 164 GB | – | – | 未試験 |

lossy な設定（Maya-S-v2、5090）: `STRATA_GLM_DEFER=1` は -19% / トークン、nll +0.063。

8 GB のカード（issue #23、RTX A2000 Laptop）: Maya-S-v2 は dense pack を IQ4_XS に再量子化すると起動します（5.6 → 3.6
GiB、手順は[VRAM の小さいカード](SMALL-VRAM.ja.md)）。そこでは 4〜4.7 tok/s で、512 GB SSD の読みが上限でした。

### Ornith-1.5-122B（Qwen3.5-MoE）

ダウンロード数は未調査。UD-Q4_K_XL は VRAM 32 GB + RAM 層 60 GB で全エキスパートが載り、SSD を読まない。

| ファイル（入手先） | サイズ | 品質 | 速さ | おすすめ |
| --- | ---: | --- | --- | --- |
| `UD-Q4_K_XL`（[unsloth/Ornith-1.5-122B-GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF)） + NextN ドラフト（`tools/nextn_pack.py`） | 79.7 GB + 1.5 GB | llama.cpp との KL 0.024（v1.3.46）。swe-bench L7: 11/16 解決（v1.3.49） | R9700 + レーン: 1 トークンずつ 41〜43、NextN でコード 59 / 文章 50（v1.3.46）、19K の後 51.0（v1.3.49）、swe L7 54.9 tok/s（v1.3.49）。レーンなし: コード 42 / 文章 39（v1.3.46） | 40〜50 tok/s を SSD なしで |
| `UD-IQ3_XXS`（unsloth） | 49.5 GB | 未計測 | ユーザー報告のみ（下の #8） | 16 GB のカード向け |
| `IQ3_XXS-B`（[zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF](https://huggingface.co/zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF)） | 51.7 GB | KL は UD-IQ3_XXS とほぼ同じ（0.138〜0.140）。swe-bench 60 問中 56 問解決 | 5090、`--model ornith` が書く設定: 116 tok/s、NextN の下書きの採用率 75%（v1.3.100） | GPU 16 GB + RAM 64 GB |

### Qwen3.8-Flash-Next（qwen4exp）

詳細、pack、ドラフターは [docs/qwen38](qwen38/README.ja.md)。KL は特に書かない限り NVFP4 教師との比較（seq 0〜9、
コンテキスト 2,048）。速さは 4K のプロンプトで 512 トークン、3 回の中央値、MTP ドラフターあり、RTX 5090（PCIe Gen4 x4）+
RAM 94 GB、iGPU レーンなし。

| ファイル（入手先） | サイズ | 品質 | 速さ | おすすめ |
| --- | ---: | --- | --- | --- |
| UD 3.1 `UD-IQ4_XS`（[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)、ベータ） | 93.7 GB | KL 0.077（同じファイルの llama.cpp は 0.076） | 32 GB: 117.0、16 GB: 80.9（v1.3.115） | 32 GB + 96 GB で品質重視 |
| UD 3.1 `UD-IQ3_XXS`（unsloth、ベータ） | 83.6 GB | KL 0.099 | 32 GB: 133.0、16 GB: 87.4（v1.3.115） | 速さと品質の釣り合い |
| UD 3.1 `UD-IQ1_M`（unsloth、ベータ） | 68.0 GB | KL 0.190 | 32 GB: 151.8、16 GB: 77.0（v1.3.115） | いちばん小さく速い |
| `16G64G`（[zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-IQ3_M-16G64G-GGUF)、ベータ） | モデルカード参照 | KL 0.083 | 32 GB: 131.3、16 GB: 96.4（v1.3.112）。DGX Spark 47.05（v1.3.113） | GPU 16 GB + RAM 64 GB |
| `12G32G` L448（[zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-E448-IQ2_M-12G32G-GGUF)、512 個中 448 個のエキスパート、ベータ） | 67.5 GB | 本ページの尺度では未計測 | 本ページの尺度では未計測 | GPU 12 GB + RAM 32 GB |
| ISTA-DASLab GSQ-RCO `IQ3_S` | 83.6 GB | llama.cpp との KL 0.026（R9700、v1.3.76） | 5090、コンテキスト 32K: `--bench` 128.5〜129.5（v1.3.87）。プロンプト 8K 2,240〜2,376、32K 3,002〜3,027（v1.3.88） | – |
| magnitudedev `NVFP4` | 127.8 GB | llama.cpp との KL 0.015 / 0.018（トークン経路 / プロンプト経路） | 5090 + 96 GB: swe 3 問 105.3、32K プロンプト 2,515（v1.3.101） | Blackwell の FP4 でプロンプト経路 |

## 2. GPU 別の参考ベンチマーク（作者の環境）

RTX 5090 の v1.3.38 までの値は **2026-10-23 の SSD 交換より前**（ORICO 約 3.4 GB/s）。5090 は 2026-10-31 に戻し、v1.3.82
以降の行は NM790 での値です。V4.1 の ISTA と UD-IQ1_S は 5090 で測り直していません。

| モデル / ファイル | 項目 | RTX 5090 + 8060S レーン | AI PRO R9700 単体（CPU レーン） | R9700 + 8060S レーン | RTX 3090（USB4）+ レーン |
| --- | --- | --- | --- | --- | --- |
| V4 0731 UD-Q2_K_XL | 1 トークンずつ | 92〜95（v1.3.21〜27） | 46.1〜46.5（v1.3.41） | 52.2〜52.9（v1.3.40） | 54.4〜56.5（v1.3.27） |
| | swe 6 問 | 104.8（v1.3.27） | 48.4、DSpark 61.9（v1.3.43） | 51.7、DSpark 67.3（v1.3.43） | 57.9（v1.3.27） |
| | swe 60 問 × 5 | 106.7（v1.3.27） | 未計測 | DSpark 70.2（v1.3.43 直前） | 未計測 |
| | prompt 8K | 1,630〜1,760（v1.3.16〜18） | 844（v1.3.41） | 866〜995（v1.3.40） | 765〜864（v1.3.27） |
| V4 0731 huihui Q2 | swe 6 問 | 91.1（v1.3.27） | 33.9（v1.3.39） | 38.1（v1.3.39） | 49.1（v1.3.27） |
| V4.1 nazeshinjite Q2 + TAIL_SKIP（KL 0.241） | 1 トークンずつ | 27.9（v1.3.31） | 未計測 | 19.7（v1.3.46） | 未計測 |
| | swe 60 問 × 5 | 50.6（v1.3.33） | 未計測 | 未計測 | 未計測 |
| V4.1 UD-IQ1_S + TAIL_SKIP（KL 0.359） | 1 トークンずつ | 45.6（v1.3.38） | 未計測 | 31.9（v1.3.46） | 未計測 |
| | swe 8 問（NM790） | 未計測 | 未計測 | 43.7（+ affinity、KL 0.3573）、DSpark 54.4（2026-10-24） | 未計測 |
| V4.1 ISTA IQ2_XXS + TAIL_SKIP（KL 0.3266） | swe 8 問（NM790） | 未計測 | 未計測 | 42.2 / 41.3、DSpark 48.3 / 47.3（2026-10-24） | 未計測 |
| V4.1 UD-IQ2_XXS、コンテキスト 64K | `--bench` の生成 / 8K prompt（DSpark なし） | 23.1 / 357（v1.3.82）、22.9 / 360（v1.3.87） | 未計測 | 未計測 | 未計測 |
| | swe 3 問、普段の設定 + DSpark | 51.0（v1.3.89） | 未計測 | 未計測 | 未計測 |
| GLM Maya-S-v2 | 1 トークンずつ | 64.7〜75.4（v1.3.27）。`--bench` 64.0〜64.6（v1.3.82〜87） | 未計測 | 34.3〜38.3（v1.3.40） | 40.5〜42.4（v1.3.27） |
| | swe 6 問 | 64.1（v1.3.27） | 25.8（v1.3.39） | 35.8（v1.3.39） | 39.7（v1.3.27） |
| | prompt 8K | 1,087〜1,113（v1.3.82〜87） | 未計測 | 未計測 | 未計測 |
| GLM EXL3 sc_2.10bpw | 1 トークンずつ | 約 53（v1.3.36） | 未計測 | 未計測 | 未計測 |
| GLM exlr8 K2 | 1 トークンずつ | 未計測 | 未計測 | 約 22.9（2026-10-24） | 未計測 |
| GLM exlr8 K3 | 1 トークンずつ | 未計測 | 未計測 | 約 11.2（2026-10-24） | 未計測 |
| Ornith UD-Q4_K_XL + NextN | コード / 文章 | 未計測 | 42 / 39（v1.3.46） | 59 / 50（v1.3.46） | 未計測 |
| | swe L7 | 未計測 | 未計測 | 54.9（v1.3.49） | 未計測 |
| Qwen3.8 ISTA IQ3_S + MTP | `--bench` の生成 | 128.5〜129.5（v1.3.87） | 未計測 | 文章 51 / コード 77（v1.3.76） | 未計測 |
| Qwen3.8 16G64G + MTP | 4K プロンプト、512 トークン | 32 GB で 131.3、16 GB で 96.4（v1.3.112） | 未計測 | 未計測 | 未計測 |

V4.1 の R9700 の 1 トークンずつの値は 8 系列の平均で、ORICO の SSD（2026-10-22）。swe 8 問は NM790 に替えた後。
R9700 単体の UD-IQ1_S は 1 トークンずつ約 22.7 tok/s（44.0 ms、v1.3.46。TAIL_SKIP の有無は記載なし）。
5090 + 3090 で層を分けると 5090 1 枚より遅い（87〜90 tok/s、v1.3.25 前後）。

### DGX Spark

レンタルの DGX Spark（GB10、共有メモリ 128 GB）、Abyss v1.3.113〜v1.3.114（2026-11-05 / 06）、ベータ
（[DGX Spark](DGX-SPARK.ja.md)）:

| モデル / ファイル | 生成 | prompt |
| --- | --- | --- |
| Qwen3.8 16G64G + MTP（v1.3.113） | 47.05 tok/s（CLI、4K、512 トークン × 3 回の中央値。なしで 31.2）。サーバー経由で 43〜54 | 8K 約 1,040 tok/s |
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS / unsloth UD-IQ2_XXS | 19〜20 tok/s（ここでは MTP ドラフターの効果なし） | 8K 730〜750 tok/s。CLI で 32K 675〜796 |
| DeepSeek V4 0731 | 21.0 tok/s、DSpark ありで 22.8（下書きの採用率 78%） | 8K 814 tok/s |
| DeepSeek V4.1 Compact JA256（207 GB、メモリに収まらない） | 約 18 tok/s | 未計測 |

同じ Spark の llama.cpp（8K prompt、コンテキスト 4K 分を埋めた状態）: V4 0731 は 20.2 tok/s / prompt 360、GLM は 19.3 / 321。
Abyss は生成が同じかそれより速く、prompt は約 2.3 倍速く読みます。既知の問題（未解決、Spark だけで出る）: サーバー経由では
32K〜121K の prompt が約 120〜130 tok/s で、CLI の 675〜796 より遅くなります。調査中です（RTX 5090 では出ません）。
v1.3.115 から、ほかのプロセスが共有メモリを 16 GB より多く使っていて、そのためにエキスパートが収まらないときは、エンジンは
起動しません（`STRATA_GLM_UMA_SHARE=1` で起動を続け、`STRATA_GLM_UMA_OTHERS_GB` で上限を変えられます）。

## 3. ユーザー報告（参考）

GitHub の issue で利用者が報告した値です。**利用者が測ったもので、作者は再現していません。** 測り方（プロンプトの長さ、
`glm stat` の値か通しの値か）は報告ごとに違うので、上の表とは直接比べられません。

| issue | PC | Abyss | モデル / ファイル | 設定 | 生成 tok/s |
| --- | --- | --- | --- | --- | --- |
| #8 | RX 9070 XT 16 GB + Ryzen 7 7800X3D + DDR5 64 GB、PCIe 4.0 NVMe、Linux（Docker、ROCm 10） | v1.3.44 | V4 0731 UD-Q2_K_XL | TAIL_SKIP なし、DSpark なし | 18.3〜18.4（温まった後）、8.8K prompt の後 16.7。prompt 133〜148 |
| #8 | 同上 | v1.3.44 / v1.3.46 | 同上 | `TAIL_SKIP=3` + `_DISK=1`（V4 での KL は未計測。V4.1 では 0.229 → 0.241） | 23.7〜24.1。同じ話題が続くと 29〜31（v1.3.46 の RAM fill で省く数 -40%、速さは同じ） |
| #8 | 同上 | v1.3.44 | GLM Maya-S-v2 IQ2_XXS | 既定 | 10.3〜10.6 |
| #8 | 同上 | v1.3.46 | Ornith UD-Q4_K_XL / UD-IQ3_XXS | `TAIL_SKIP=3` + `_DISK=1`（KL 未計測）、NextN（v1.3.49 で直したバグのある版）、iGPU なし | 文章 20.8 / 21.9、コード 25〜29 / 34〜39 |
| #11 | 同じ PC（7800X3D の iGPU は BIOS で無効、PCIe Gen5 x16）、Linux（ROCm 10 のコンテナ） | v1.3.53 | V4.1 1017 ISTA GSQ-RCO IQ2_XXS、128K、`RAM_GB=44` | `TAIL_SKIP=3` + `_DISK=1`。レーン有効（不具合）/ `STRATA_GLM_IGPU=0` | 初回 18.3 / 26.3。同じ話題 22.9〜24.0 / 31.3〜31.5。新しい話題 17.7〜17.9 / 18.7〜20.9。10K のプロンプト後 18.0 / 20.3 |
| #11 | 同上 | v1.3.53 | Ornith UD-IQ3_XXS、128K、思考なし | レーン有効（不具合）/ `STRATA_GLM_IGPU=0` | 25.9〜28.0 / 39.1〜39.3 |
| #11 | 同上、ROCm 7.2.1 のコンテナ（`ABYSS_OLD_ROCM=1`） | v1.3.53 | Ornith UD-IQ3_XXS | `STRATA_GLM_IGPU=0`（7.2.1 ではレーン有効だと全リクエストが失敗。v1.3.54 で修正） | 32〜35（短い回答） |
| #5 | RX 7900 XT 20 GB + i7-14700K + DDR4 128 GB、Windows 11（TheRock 7.14.1） | v1.3.30 → v1.3.46 | V4 0731 UD-Q2_K_XL、64K | `TAIL_SKIP=0` | 13.71 → 12.56 |
| #5 | 同上 | v1.3.30 → v1.3.46 | 同上 | `TAIL_SKIP=1`、`LANE=8`、`PIN=0-15`（KL 未計測、品質の変化は大きい） | 28.20 → 32.86 |
| #5 | 同上 | v1.3.30 → v1.3.46 | 同上、256K | `TAIL_SKIP=4`（KL 0.0226、5090 で計測） | 17.77 → 16.40 |
| #5 | 同上 | v1.3.30 → v1.3.46 | V4.1 nazeshinjite Q2、RAM 層 85 GB | `TAIL_SKIP=4` + `_DISK=1`（KL 0.235） | 10.55 → 10.12 |
| #9 | 同上 | v1.3.46 + 利用者自身のパッチ（本家に入っていない） | V4 0731 UD-Q2_K_XL | `TAIL_SKIP=1` / `=4` / `=0`（=1 の KL 未計測、=4 は 0.0226） | 約 53〜55 / 約 35 / 約 18（パッチ前 約 36.8 / 16.5 / 10.7） |
| #9 | 同上 | 同上 | V4.1 nazeshinjite Q2 | `TAIL_SKIP=2` + `_DISK=1`（KL 未計測）/ `TAIL_SKIP=0` | 約 44.0 / 約 35.8 |
| #1 | AI PRO R9700 + i5-14400F + RAM 128 GB、Ubuntu（ROCm 7.0、v1.3.42 から ROCm 10） | v1.3.20 | V4 0731 UD-Q2_K_XL | 既定 | 28.4〜31.1、温まると 36.6 |
| #1 | 同上 | v1.3.31 | 同上、393K コンテキスト | 既定 | 291K の prompt を 279 tok/s で読み、生成 19.5 |
| #1 | 同上 | v1.3.35 / v1.3.37 / v1.3.42 | V4.1 nazeshinjite Q2、RAM 層 80 GB | `TAIL_SKIP=3` + `_DISK=1`（KL 0.241） | 24〜28 / 30.5 / 22〜25。8K prompt 42〜48 → 60（v1.3.42） |
| #23 | RTX A2000 Laptop 8 GB + RAM 64 GB、PCIe Gen4 x8 | v1.3.82 より前 | GLM Maya-S-v2（IQ4_XS の dense pack）、Ornith IQ3_XXS-B | route より小さいエキスパートの置き場（8 スロット中 6） | GLM 4〜4.7（512 GB の SSD の読みが上限）。v1.3.82 でこうした置き場でも起動し、prompt をまとめて読めるようになった（[VRAM の小さいカード](SMALL-VRAM.ja.md)） |
| #31 | RTX 4090 + RAM 96 GB、Windows | v1.3.107 より前 | – | `STRATA_GLM_RAM_GB=72` | –（約 20K トークンからの prompt が失敗し、約 40K からエンジンが落ちた。v1.3.107 で、CPU 分担の pinned バッファが確保できないときは PCIe に回すようにした） |
| #35 | 16 GB の GeForce 2 枚 | v1.3.109〜v1.3.110 | Qwen3.8-Flash-Next、GPU 2 枚への分割（ベータ） | `--layer-split 32` と 1 枚の比較 | –（長いリクエストで分割が 22% 速い。`auto` はまだ 1 枚を選ぶので、手で分ける） |
| #37 | DGX Spark | v1.3.113〜v1.3.114 | – | `./abyss.sh --check`、`--gpu-tune` | –（短くて正しい日本語の答えが `--check` で不合格になった。v1.3.114 で修正。`--gpu-tune` は v1.3.115 から手でビルドしたエンジンを見つける） |

#8 の行と #11 の「レーン有効」の行は、iGPU レーンが誤って 9070 XT 自身の上で動いた状態で測ったもの（iGPU が見えないとき、
レーンがエンジンのカードに落ちていた。issue #11、v1.3.54 で修正）。レーンを切ると V4.1 で約 30%、Ornith で 40〜50% 速い。

`./abyss.sh --gpu-tune` の報告（dense の Q6_K GEMV、1 回の合計 µs。小さいほど速い）:

| issue | GPU | Abyss | 4 warps / 1 行 1 warp | エンジンの選択 |
| --- | --- | --- | --- | --- |
| #10 | RX 9070 XT 16 GB（gfx1201）、Ryzen 7 7800X3D、RAM 62 GB | v1.3.53 | 868.0 / 537.8（大きい行列で 590 GB/s 前後） | 1 行 1 warp（最速と一致、変更なし） |

### 報告から言えること

- 16 GB のカードと 64 GB の RAM（#8）でも、V4 0731 は TAIL_SKIP なしで約 18 tok/s（レーンの不具合込み）。V4.1 の ISTA IQ2_XXS は
  TAIL_SKIP=3 で 19〜31 tok/s（#11、レーンなし）。
- ROCm 7.2.1 でも動く（同じ PC で Ornith 32〜35 tok/s、ROCm 10 では約 39。#11）。setup は ROCm 10 を求めるので、
  `ABYSS_OLD_ROCM=1` で通す。
- 128 GB の RAM（#1、#5）なら V4.1 の Q2 も動く。速さは SSD と RAM 層の大きさで決まる。
- `TAIL_SKIP=1` / `=2` の速さは KL を測っていない設定の値です。品質を比べるときは `=3`（V4.1、`_DISK=1` と一緒）か `=4` 以上を。
