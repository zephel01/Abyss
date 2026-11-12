# VRAM の小さいカード（8〜12 GB）

[English](SMALL-VRAM.md)

Abyss の動作確認は 16 GB + メモリ 64 GB（と 32 GB のカード）でしています。8 GB のノート PC 用 GPU でも同じモデルは
動きますが、余地が違います。dense の重み（attention、共有エキスパート、埋め込み）で 5〜6 GB、KV キャッシュで少し、
残りがエキスパートの置き場（pool）になり、層あたり数個分しかありません。このページは、その少ない置き場で Abyss が
何をするか（v1.3.82 から、issue #23）、拒否・警告の行の読み方、そして GLM-5.3-Flash で一番効く 1 手 — dense pack を
IQ4_XS に軽くする（コマンド 1 本）— をまとめたものです。

これまでの実測（issue #23、RTX A2000 Laptop 8 GB、PCIe Gen4 x8、Ryzen 9 7940HX、DDR5 64 GB、Abyss v1.3.77）:
Ornith-1.5-122B IQ3_XXS-B は decode 15.8 tok/s、prompt は 16 tok/s（1 トークンずつ）。GLM-5.3-Flash Maya-S-v2 は
dense pack を IQ4_XS に再量子化するまで起動せず、再量子化後は 512 GB SSD で 4〜4.7 tok/s（SSD 律速: RAM 層に
96 GB のうち 40 GB）。

v1.3.82 で、RTX 5090 に報告者と同じ大きさの置き場を与えて測った値（`STRATA_GLM_POOL_GB`、最後の節。PCIe Gen4 x4、
メモリ 94 GB）、`./abyss.sh --bench`:

| | v1.3.81 の動き | v1.3.82 |
|---|---|---|
| Ornith-1.5-122B IQ3_XXS-B、置き場 1.45 GB（層あたり 8 個）— prompt 2K / 8K | 46 tok/s（1 トークンずつ） | **356 / 347 tok/s**（768 トークンの chunk） |
| 同 — decode | 48.3 tok/s | 45.5 tok/s（実行ごとの幅 42〜49） |
| 同 — 品質（16 GB で作った参照との KL） | 0.0073 | 0.0073 |
| GLM-5.3-Flash Maya-S-v2（dense Q6_K）、置き場 1.2 GB（層あたり 4 個） | 起動拒否（`not enough VRAM`） | **起動する**。decode 9.1 tok/s（SSD 律速、RAM 層 40 GB）、KL 0.039（32 GB での実行は 0.036〜0.044） |

ノート PC のカードはリンク（x8）も CPU も違うので、同じ置き場でもこれより下がります。見てほしいのは prompt の比
（7.7 倍）です。

## 目次

- [小さいカードで Abyss がすること](#小さいカードで-abyss-がすること)
- [ログの行](#ログの行)
- [GLM-5.3-Flash: IQ4_XS の dense pack](#glm-53-flash-iq4_xs-の-dense-pack)
- [8 GB で Ornith-1.5-122B](#8-gb-で-ornith-15-122b)
- [設定](#設定)
- [大きいカードで試す](#大きいカードで試す)

## 小さいカードで Abyss がすること

層あたり数個の置き場しかないとき、v1.3.82 で 3 つ変わりました。

1. **置き場は route より小さくてよい。** GLM-5.3-Flash と Ornith は毎層 8 個のエキスパートを通します。以前は
   層あたり 8 個分の置き場がないとエンジンが起動を拒否していました（"not enough VRAM for the expert pool"）。
   いまは 4 個以上あれば起動します。VRAM にないエキスパートは予備（spare）の枠に入るか、空きがなければ自分用の
   scratch 枠に入り、CPU レーンの分は RAM から計算します — どれも route 全部が入る必要はありません。issue #23 の
   報告者は 8 個中 6 個の置き場で動かし、出力はエンジン自身の参照と完全一致（KL 0）でした。ただしこの置き場は
   毎トークンほとんど外すので遅く、下の dense pack の方が本筋です。4 未満は今も拒否し、VRAM が何 MB 足りないかを
   言います（次の節）。
2. **prompt は 1 トークンずつでなく、まとめて読む。** prompt 経路はバッファをエキスパート置き場から借ります。以前は
   各層に「route 1 本分 + 予備 + 2」（13 個）を残していたので、9 個の置き場は何も貸せず、prompt は 1 トークンずつ
   — issue #23 の 16 tok/s です。いまは、それでは貸せない置き場は prompt の間だけ層あたり 4 個を残して残りを
   貸します（9 個なら 5 個。Ornith で約 800 MB、約 2,000 トークンの chunk）。prompt は単独で走り、置き場に何が
   あろうと層のほぼ全エキスパートを流し込むので、貸した常駐分の損は小さく、終わった後は decode のミスが尾を埋め
   直します（1.5 GB、1 秒未満）。issue #23 の 2,048 トークン prompt での試算: エキスパートを chunk ごとに層あたり
   1 回流す（PCIe Gen4 x8 で 42 GB、約 3 秒）+ GPU 自身の行列計算で、16 tok/s に対して 150〜400 tok/s 程度。
   v1.3.137 からは、本来の残しで貸せる chunk が短い（4,096 トークン未満）ときも、小さい残しでその 2 倍以上の chunk
   を貸せるなら小さい残しにします（`STRATA_GLM_PREFILL_SMALL_BELOW`）。また、実際にできた置き場が最初の見積もりより
   小さかったとき（間に下書きモデルや context が VRAM を取ったとき）は、実際の置き場に合わせて chunk を選び直します。
   16 GB のカードで Ornith-1.5-397B 16G64G: 層あたり 19 個のうち 4 個しか貸せなかった置き場（chunk 1,024、4K の
   prompt 約 70 tok/s）が 15 個を貸すようになり（chunk 12,288、約 210 tok/s。8K は 65 -> 390）、何も貸せなかった
   15 個の置き場（4K の prompt を 1 トークンずつ約 400 秒）は約 250 tok/s になりました。
3. **CPU がリンクよりずっと速ければ、RAM 層のエキスパートは全部 CPU レーンへ。** CPU レーンと PCIe の分担は起動時に
   測って決めます（issue #23 では `an expert 0.055 ms on the CPU vs 0.282 ms over PCIe`）。この分担は 8 個のうち
   1 個をリンクに残しますが、その箱では全部 CPU の方が 10% 速かった（`--deep` の分担掃引: share 0 が 18.5 tok/s、
   エンジンの分担 16.8）。両レーンが同じ DRAM を読み、リンクのコピーは層のエキスパートより前に GPU のストリームに
   並ぶからです。CPU がリンクの 4 倍（`STRATA_GLM_LANE_LEAN`）速ければ、置き場が落ち着いた（ヒット率が伸びなくなった）
   時点で RAM 層のエキスパートを全部 CPU に回します。それまでは測った分担のまま（prompt 後のミスで置き場を埋め直す
   ため、issue #14）、ヒット率が落ちたら戻します。分担がもともと全部 CPU のカード（Gen4 x4 / OCuLink）と、リンクが
   CPU に近いカード（Gen5 x16 の R9700: 0.311 対 0.499 ms）は、これまで通りです。

route が入り、chunk を貸せる置き場のカードでは何も変わりません。3 つの規則はその下でだけ働きます。

## ログの行

```
glm fast: CUDA0 VRAM before the expert pool: 5.72 of 7.71 GB in use - dense weights 5.17, state/KV (8192 ctx) 0.00, ...
glm fast: CUDA0 layers [0,48) expert pool 1.45 GB, 9-9 slots/layer (432 total)
```

VRAM の内訳です。dense weights は pack の attention / 共有エキスパート / 埋め込み（下の IQ4_XS pack で
GLM-5.3-Flash は約 2 GB 小さくなります）、state/KV は `--max-context` で増え（`--kv q4_0` で半分以下）、置き場は
残りから予約分（`STRATA_GLM_RESERVE_MB`、12 GB 以下の CUDA カードで 700 MB）を引いたものです。

```
glm fast: not enough VRAM for the expert pool: 3 slots per layer, 4 at least (170 MB short; a whole route's 8 slots:
  850 MB short) - a smaller --max-context, --kv q4_0 or a lighter dense pack (tools/requant_dense.py IQ4_XS,
  docs/SMALL-VRAM.md) frees it
```

拒否の行です。前の数字は起動に足りない分、後ろは route 1 本が丸ごと入るのに足りない分（そこから先は毎トークンの
ミスが減ります）。3 つのどれでも VRAM が空きます: `--max-context 8192`（65536 でなく。KV キャッシュ）、`--kv q4_0`
（KV キャッシュを fp16 の 4 分の 1 に）、dense pack（一番大きく、GLM-5.3-Flash で約 2 GB）。

```
glm fast: CUDA0 the expert pool holds 6 slots a layer, below a route's 8: the rest of every route runs from the
  scratch slots and the CPU lane (340 MB more VRAM would hold a route; docs/SMALL-VRAM.md)
glm prefill: CUDA0 the expert pool (9 slots a layer) can lend no prompt chunk while every layer keeps 13 slots - it
  keeps 4 while a prompt runs (small VRAM; STRATA_GLM_PREFILL_KEEP)
glm prefill: CUDA0 chunks of 2048 tokens (--prefill auto, up to 32768; ...)
glm fast: CUDA0 CPU lane: the CPU does an expert 5.1x faster than the link - every RAM-tier expert goes to the CPU
  once the pool settles (windows of 128 token steps), the split below until then and when the VRAM hits fall
  (STRATA_GLM_LANE_LEAN=0 keeps the split)
```

上の 3 つの規則が働いたときの行です。それでも chunk を貸せないときは、最小の chunk に何が要り、置き場が何を
貸せるかを言います。

```
glm prefill: CUDA0 the expert pool (4 slots a layer, 4 kept) can lend no prompt chunk (--prefill auto): a 256-token
  chunk needs 120 MB, the pool can lend 0 MB - 120 MB of VRAM short; a smaller --max-context, --kv q4_0 or a lighter
  dense pack (docs/SMALL-VRAM.md) frees it - token by token
```

## GLM-5.3-Flash: IQ4_XS の dense pack

Maya-S-v2 の dense テンソルは Q6_K で、GGUF 内 5.6 GiB、エンジンの表を含めて VRAM 5.97 GB です。IQ4_XS に
再量子化すると 3.6 GiB（issue #23 の報告者は Q5_K 版から 4.68 → 3.61 GiB、VRAM の行は
`dense weights 5.97 -> 4.92`）になり、層あたり 4 個だった置き場が 8 個 — route 1 本分 — になります。これらの
テンソルを IQ4_XS にした品質の差は Q6_K pack と比べてまだ測っていません（報告者の実行は自分自身と決定的に一致、
KL 0）。DeepSeek V4 で attention を Q5_K にしたときは KL 0.006 → 0.012 でした
（[マニュアル](MANUAL.ja.md)の「軽い dense pack」）。

対象のテンソル: attention の射影（`attn_output`、`attn_q`、`attn_k`、`attn_v`、`attn_q_a`、`attn_q_b`、
`attn_kv_a_mqa`）、共有エキスパート（`ffn_*_shexp`）、dense FFN の 3 層（層 0〜2 の `ffn_down/gate/up`）。
そのまま残すもの: `token_embd` と `output`（Q6_K）、`attn_k_b` / `attn_v_b`（Q6_K。3 次元テンソルで、再量子化
ツールが触ってはいけないもの）、F32 の norm とルーター、Q8_0 の再帰 / indexer。ルーティングされるエキスパートは
GGUF のまま手を付けません。

`tools/requant_dense.py` には llama.cpp のビルドの `libggml-base.so`（量子化器。`llama-quantize` 自体は imatrix の
ない i-quant モデルを拒否します）と gguf-py（Abyss の `third_party/llama.cpp/gguf-py`）が要ります。96 GB の shard
を元のディレクトリの隣に書き直し（NVMe で約 15 分）、続けて `iq_pack.py` で pack を作ります:

```sh
cd ~/llm/apps/Abyss
M=~/llm/apps/Maya-data/models
GGUF_PY=$PWD/third_party/llama.cpp/gguf-py LIBGGML=~/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so \
  .venv/bin/python -I tools/requant_dense.py $M/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  $M/Maya-S-v2-IQ2_XXS-iq4dense iq4_xs --from=q6_k \
  'blk\.\d+\.attn_output\.weight' 'blk\.\d+\.attn_[qkv]\.weight' 'blk\.\d+\.attn_q_[ab]\.weight' \
  'blk\.\d+\.attn_kv_a_mqa\.weight' 'blk\.\d+\.ffn_(down|gate|up)_shexp\.weight' 'blk\.[0-2]\.ffn_(down|gate|up)\.weight'
.venv/bin/python tools/iq_pack.py --gguf $M/Maya-S-v2-IQ2_XXS-iq4dense/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  --out $M/Maya-S-v2-IQ2_XXS-iq4dense/pack --compat-bf16
cp -r $M/Maya-S-v2-IQ2_XXS/pack/tokenizer $M/Maya-S-v2-IQ2_XXS-iq4dense/pack/
```

（`abyss.py --repack` は知っているモデル名しか受けないので、`iq_pack.py` を直接呼びます。`--from=q6_k` は元の型の
指定で、言わなければ Q8_0 のテンソルしか再量子化しません。）サーバーは止めて実行してください。新しいディレクトリに
96 GB 要ります。新しい pack が動いたら元は消して構いません（報告者は場所を空けるため中間ファイルを消しました）。

次に設定ファイル（`abyss-glm.json`、またはインストーラが書いたもの）を新しい pack に向け、文脈長と KV キャッシュを
カードに合わせます:

```json
  "args": ["--glm-pack", "/home/you/llm/apps/Maya-data/models/Maya-S-v2-IQ2_XXS-iq4dense/pack",
           "--max-context", "8192", "--kv", "q4_0", "--prefill", "auto"],
  "tokenizer": "/home/you/llm/apps/Maya-data/models/Maya-S-v2-IQ2_XXS-iq4dense/pack/tokenizer",
```

`./abyss.sh --bench` で測れます（`bench/ref` との品質確認で、IQ4_XS pack が Q6_K と同じトークンを書くかが分かり
ます）。品質は `.venv/bin/python tools/bench_ref.py <設定ファイル> --tokens 512 --k 32 --check` でエンジン自身の
参照に対する nll / top-1 が出ます。

512 GB の SSD ではモデルの 96 GB がシステムと同居し、RAM 層（64 GB のうち約 40 GB）にはエキスパートの半分も
入りません。速さは SSD の読みで決まります（issue #23 で 1 トークン 44 回、176 ms）。ここでは GPU より、速い SSD か
2 枚目の SSD（[SSD2](SSD2.ja.md)）の方が効きます。

## 8 GB で Ornith-1.5-122B

[zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF](https://huggingface.co/zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF)
（51.7 GB）は再量子化なしで 8 GB + 64 GB で動きます。dense の重みは 5.17 GB、置き場は層あたり 9 個、モデル全体が
RAM 層に入ります（温まれば SSD は読みません）。issue #23 の RTX A2000 Laptop 8 GB での数値: v1.3.77 で decode
15.8 tok/s、prompt 16 tok/s。v1.3.82 では prompt がまとめて読まれ（同じ置き場で 7.7 倍、冒頭の表）、置き場が落ち着いたら CPU レーンが RAM 層の
エキスパートを全部受け持つ（分担掃引の 18.5 tok/s）ので、`STRATA_GLM_PCIE_SHARE=0` を手で書く必要はなくなります。
設定は `--max-context 8192`（このモデルの KV キャッシュは小さく、16K も入ります）。

## 設定

| 設定 | 既定 | 働き |
|---|---|---|
| `STRATA_GLM_POOL_FLOOR` | 4 | エキスパート置き場の最小（層あたりの枠数。予備 + 常駐 1）。これ未満ではエンジンは起動を拒否し、VRAM が何 MB 足りないかを言う。v1.3.82 より前は route 1 本分（8）が床だった |
| `STRATA_GLM_PREFILL_KEEP` | 4 | 本来の残し（route + 予備 + 2）では chunk を貸せない置き場で、prompt の間に各層が残す枠数。大きくすると prompt 中も常駐が残る代わりに chunk が小さくなる |
| `STRATA_GLM_PREFILL_SMALL_BELOW` | 4096 | 本来の残しで貸せる chunk がこのトークン数より短く、小さい残しでその 2 倍以上を貸せるときも、小さい残しにする。`0` で、本来の残しで chunk を貸せないときだけ（v1.3.136 の動き） |
| `STRATA_GLM_LANE_LEAN` | 4 | CPU レーンがエキスパート 1 個を PCIe のこの倍数速くこなすなら、置き場が落ち着いた時点で RAM 層のエキスパートを全部 CPU に回す。`0`: v1.3.82 より前と同じ測った分担だけ。手で書いた `STRATA_GLM_PCIE_SHARE` / `_LATE` / `STRATA_GLM_CPU_PLAN` が優先 |
| `STRATA_GLM_VRAM_GB` | オフ | この GB のカードのように振る舞う: 置き場はそのカードが持てる分だけ（すでに確保した文脈、dense、状態を数える）。このページの規則を大きいカードで試すためのもの |
| `STRATA_GLM_RESERVE_MB` | 700（12 GB 以下の CUDA）、1536（それより大きい CUDA、HIP） | 置き場の横に空けておく VRAM。issue #23 は CUDA で 64 にして置き場が約 0.6 GB 増えた |

## 大きいカードで試す

設定ファイルの `"env"` に `STRATA_GLM_VRAM_GB=8` と書くと、16 GB や 32 GB のカードでも置き場を 8 GB のカードの
ように決めます（prompt 経路も同じ上限で chunk を決めます）。dense の重みと KV キャッシュは普段通り確保されるので、
上限はその合計 + 予約分より大きい必要があります。ログの `VRAM cap` の行にエキスパートに残った分が出ます:

```
glm fast: CUDA0 VRAM cap 8.0 GB: 5.72 GB already in use, 1.60 GB left for experts
```

大きいデスクトップのカードは CUDA の文脈・MTP・scratch がノート PC より大きく（RTX 5090 + Ornith で 1.41 GB、8 GB の
報告から読める値は約 1 GB）、`STRATA_GLM_VRAM_GB=8` では本物の 8 GB カードより置き場が小さくなります — Ornith で
層あたり 9 個のところが 4 個になり、床の置き場は prompt の chunk を貸せません。報告を再現するときは置き場の大きさを
直接与えます: `STRATA_GLM_POOL_GB` = 報告者のログの `expert pool N GB`（冒頭の表は Ornith 1.45、GLM-5.3-Flash 1.2）。
