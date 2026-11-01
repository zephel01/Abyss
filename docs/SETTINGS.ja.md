# 設定ファイル

[English](SETTINGS.md)

Abyss の設定ファイルについては、このページにまとめています。初めて使う人向けに、次の順で説明します。
1. どのファイルがどこにあるか
2. 変えたいことはどこを書き換えるか
3. 書き換えたあと、どうすれば効くか

## 目次

- [セットアップと設定の違い](#セットアップと設定の違い)
- [変えたいこと別の早見表](#変えたいこと別の早見表)
- [ファイルの場所](#ファイルの場所)
- [いま使っている設定ファイルを見つける](#いま使っている設定ファイルを見つける)
- [書き換える手順](#書き換える手順)
- [設定ファイルの中身（実例）](#設定ファイルの中身実例)
- [すべての項目](#すべての項目)
- [Abyss が自分で書き換える項目](#abyss-が自分で書き換える項目)
- [そのほかのファイル](#そのほかのファイル)

## セットアップと設定の違い

| | セットアップ | 設定 |
| --- | --- | --- |
| いつ | 最初に 1 回（モデルを足すときにもう 1 回） | そのあと、必要なときにいつでも |
| すること | PC の確認、エンジンのビルド、モデルのダウンロード、pack の作成、**設定ファイルと起動スクリプトを書き出す** | 書き出された設定ファイルを開いて書き換え、サーバーを再起動する |
| 説明 | [MANUAL: まとめてセットアップ](MANUAL.ja.md#まとめてセットアップ) | このページ |

セットアップが最後に書き出すファイルが、このページで扱う「設定ファイル」です。セットアップをもう一度実行しても、すでにある
設定ファイルはそのまま残ります（書き直すのは `--rewrite-config` を付けたときだけです）。

## 変えたいこと別の早見表

ここでの「設定ファイル」は、Abyss フォルダーにある `abyss-<名前>.json` です。

| 変えたいこと | 書き換える場所 |
| --- | --- |
| コンテキストの長さ | 設定ファイルの `"args"` の `"--max-context"` の次の数字。ダッシュボードから変えても、ここに保存されます |
| 答えのランダムさ（temperature など） | 設定ファイルの `"sampling"` |
| 思考の既定の深さ | 設定ファイルの `"reasoning_effort"` |
| ポート番号 | 設定ファイルの `"port"` **と** 起動スクリプト `run-abyss-<名前>.sh` の `"--port"` の両方 |
| 他の PC から接続する | 設定ファイルに `"host": "0.0.0.0"` と `"api_key": "<合言葉>"`（必ず両方） |
| 使う GPU | 設定ファイルの `"gpu"`（AMD は `"gpu_pci"` も） |
| 無検閲化 | 設定ファイルの `"uncensor"`（[UNCENSOR](UNCENSOR.ja.md)） |
| エンジンの細かい調整（RAM の使い方など） | 設定ファイルの `"env"`（変数の一覧は [MANUAL: 設定項目](MANUAL.ja.md#設定項目)） |
| API が返すモデル名 | 設定ファイルの `"model_name"` |

どれも、書き換えたあとに**サーバーの再起動**が必要です（[書き換える手順](#書き換える手順)）。

## ファイルの場所

`git clone` した場所を `~/llm/apps/Abyss` とした場合の配置です。セットアップは、その隣に `Abyss-data` と `Abyss-tools` を作ります。

```
~/llm/apps/
├── Abyss/                                  プログラム本体（git で更新する）
│   ├── abyss-dsv4-q2_k_xl.json             ★ 設定ファイル（setup-deepseek.sh が書き出す）
│   ├── run-abyss-dsv4-q2_k_xl.sh           　 その起動スクリプト
│   ├── abyss-dsv4-q2_k_xl.log              　 エンジンのログ
│   ├── abyss-dsv4-q2_k_xl.shared-settings.json   チャット画面の共有設定（あれば）
│   ├── maya-<モデル>.json                   ★ 設定ファイル（GLM-5.3-Flash、./abyss.sh が書き出す）
│   ├── run-maya-<モデル>.sh                 　 その起動スクリプト
│   └── build/strata                         エンジン（AMD は build-hip/strata）
├── Abyss-data/models/...                   モデルのファイルと pack（数十〜数百 GB）
└── Abyss-tools/
    ├── abyss.sh                            起動・停止・状態確認（start | stop | restart | status | log | models）
    └── logs/
        ├── server-dsv4-q2_k_xl.log         サーバーのログ（abyss.sh log で見る）
        └── last-model                      最後に start したモデルの名前
```

ファイル名の `<名前>` は、セットアップで決まります。

| セットアップ | 設定ファイル | 起動スクリプト | 既定のポート |
| --- | --- | --- | --- |
| `./setup-deepseek.sh`（DeepSeek V4 / V4.1 など） | `abyss-<名前>.json`。名前の既定は `dsv4-<量子化>`（例: `abyss-dsv4-q2_k_xl.json`）。`--name` で変えられる | `run-abyss-<名前>.sh` | 8082 |
| `./abyss.sh`（GLM-5.3-Flash） | `maya-<モデル>.json`。AMD では `maya-<モデル>-hip.json` | `run-maya-<モデル>.sh`（Windows は `.bat`） | 8080 |
| 手で作る（[MANUAL: 設定と起動](MANUAL.ja.md#設定と起動)） | 好きな名前。Abyss フォルダーに置く | なし（`serve/server.py --config <file>` で起動） | 自分で指定 |

モデルを複数入れると、モデルごとに設定ファイルと起動スクリプトが 1 組ずつできます。

## いま使っている設定ファイルを見つける

起動スクリプトと設定ファイルは、名前で対応しています。`run-abyss-X.sh` が使うのは `abyss-X.json` です。

```sh
../Abyss-tools/abyss.sh models
```

表示の例:

```
* dsv4-q2_k_xl           ポート 8082
> dsv41-iq2xxs           ポート 8082
(* 起動中  > 最後に start したモデル。start / restart でモデル名を省くとこれを使う)
```

この例で起動中なのは `dsv4-q2_k_xl` なので、設定ファイルは `~/llm/apps/Abyss/abyss-dsv4-q2_k_xl.json` です。

確実に知りたいときは、起動スクリプトの中を見ます。`--config` の次に書いてあるのが、その設定ファイルです。

```sh
grep -o -- '--config" "[^"]*' ~/llm/apps/Abyss/run-abyss-dsv4-q2_k_xl.sh
```

## 書き換える手順

1. **バックアップを取る**: `cp abyss-dsv4-q2_k_xl.json abyss-dsv4-q2_k_xl.json.bak`
2. **テキストエディターで開いて書き換える**。JSON なので、次の点に気をつけてください。
   - 文字列は `"` で囲む（数字や `true` / `false` は囲まない）
   - 項目と項目の間に `,` を入れる。最後の項目の後ろには `,` を付けない
   - `"args"` の中は、オプションとその値を別々の文字列にする（`"--max-context", "65536"`）
3. **サーバーを再起動する**: `../Abyss-tools/abyss.sh restart`（GLM の場合は止めてから `./run-maya-<モデル>.sh`）
4. **ログで確かめる**: `../Abyss-tools/abyss.sh log` でサーバーのログ（`Abyss-tools/logs/server-<名前>.log`）が見られます。
   JSON の書き間違いや、起動しなかった理由はここに出ます。エンジンの詳しいログは、設定ファイルの `"log"` に書いたファイル
   （`Abyss/abyss-<名前>.log`）にあります。

動かなくなったら、手順 1 のバックアップに戻してください。

## 設定ファイルの中身（実例）

`./setup-deepseek.sh` が書き出す設定ファイルの例です（AMD の GPU、DSpark ドラフターあり）。

```json
{
 "exe": "/home/you/llm/apps/Abyss/build-hip/strata",
 "args": ["--glm-pack", "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack",
          "--max-context", "65536", "--prefill", "auto"],
 "cwd": "/home/you/llm/apps/Abyss",
 "tokenizer": "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack/tokenizer",
 "model_name": "deepseek-v4-flash",
 "gpu": [0],
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082,
 "log": "/home/you/llm/apps/Abyss/abyss-dsv4-q2_k_xl.log",
 "backend": "hip",
 "lib_dirs": ["/opt/rocm/lib"],
 "gpu_pci": ["0000:c7:00.0"],
 "env": {"STRATA_GLM_PREFILL_SUB": "1024", "STRATA_DS4_DSPARK": "/home/you/llm/apps/Abyss-data/models/bleysg/..."}
}
```

## すべての項目

**よく使う項目**

| 項目 | 意味 | 例 |
| --- | --- | --- |
| `exe` | エンジンの場所 | `".../Abyss/build/strata"`（AMD は `build-hip/strata`） |
| `args` | エンジンに渡すオプション。`--glm-pack` に pack のフォルダー、`--max-context` にコンテキスト長（トークン数）、`--prefill auto` はプロンプトをまとめて読む大きさ | 上の実例 |
| `cwd` | エンジンを動かすフォルダー | Abyss フォルダー |
| `tokenizer` | pack の中の `tokenizer` フォルダー（チャットテンプレートが入っている） | `".../pack/tokenizer"` |
| `model_name` | API が返すモデル名 | `"deepseek-v4-flash"` |
| `port` | サーバーのポート。起動スクリプトの `--port` と合わせる | `8082` |
| `log` | エンジンのログの書き出し先 | `".../abyss-<名前>.log"` |
| `sampling` | リクエストが指定しないときの既定値。`temperature`、`top_p`、`top_k`、`min_p`、`presence_penalty`、`repetition_penalty`、`frequency_penalty`、`penalty_last_n`、`seed`。リクエストに書いた値が優先 | `{"temperature": 1.0, "top_p": 0.95}` |
| `reasoning_effort` | 思考の既定の深さ（`none`、`low`、`high`、`max`） | `"high"` |
| `thinking_budget` | 思考の上限トークン数（既定 32768、0 で上限なし） | `16384` |
| `env` | エンジンの細かい設定（環境変数）。一覧は [MANUAL: 設定項目](MANUAL.ja.md#設定項目) | `{"STRATA_GLM_RAM_GB": "60"}` |
| `uncensor` | 無検閲化のファイル（[UNCENSOR](UNCENSOR.ja.md)） | `".../dsv4.abl"` |

**GPU**

| 項目 | 意味 | 例 |
| --- | --- | --- |
| `gpu` | 使う GPU の番号。複数書くと層を分けて載せる（1 枚目が主） | `[0]`、`[0, 1]` |
| `layer_split` | 複数 GPU のとき、2 枚目以降が受け持つ最初の層。既定は `"auto"` | `"18"`、`"16,32"` |
| `backend` | `"hip"` で AMD。省くと NVIDIA | `"hip"` |
| `gpu_pci` | AMD: GPU の PCI アドレス。再起動で GPU の番号が変わっても同じカードを使うため（セットアップが書く） | `["0000:c7:00.0"]` |
| `lib_dirs` | AMD: ROCm のライブラリの場所（セットアップが書く） | `["/opt/rocm/lib"]` |

**接続とセキュリティ**

| 項目 | 意味 | 例 |
| --- | --- | --- |
| `host` | 待ち受けるアドレス。既定 `127.0.0.1`（この PC からだけ）。`0.0.0.0` で他の PC からも接続できる | `"0.0.0.0"` |
| `api_key` | `/v1/*` に必要な合言葉（`Authorization: Bearer ...` か `x-api-key`）。カンマ区切りで複数。`host` を開けるときは必ず設定 | `"my-secret"` |
| `allow_shutdown` | `false`: `POST /api/shutdown`（エンジン暴走時などに別のプログラムからサーバーとエンジンを止める）を断る。既定は有効で、`api_key` を送れば可、キーが無ければこの PC からだけ可。Tailscale 越しなら `api_key` を設定する（[MANUAL](MANUAL.ja.md#使い方)） | `false` |
| `fit_max_tokens` | `true`: `max_tokens` が残りのコンテキストを超えたら縮めて受ける（既定は 400 で断る） | `true` |
| `mcp_servers` | チャット画面から使う MCP サーバー（Claude Desktop と同じ形式） | `{"name": {...}}` |
| `installer` | セットアップが選んだもの（モデルと量子化。Monitor の表示名用）。セットアップが書き、判断には使わない | `{"quant": "UD-IQ2_XXS"}` |

**画像（セットアップが入れるのは GLM-5.3-Flash。DeepSeek V4.1 は手で設定、下記）**: `vision` に画像エンコーダーの設定が入ります（`exe`、`mmproj`、`model`、`gpu`、`resident`、
`threads`、`max_tokens`、`no_flash_attn`）。`./abyss.sh` が書くので、通常は触りません。

知らない項目は無視されます。ただし `sampling` の値が間違っていると、サーバーは起動しません（設定の書き間違いで答え方が
黙って変わらないようにするためです）。

## Abyss が自分で書き換える項目

次の操作をすると、設定ファイルの一部が書き換わります。ほかの項目はそのまま残ります。

| 操作 | 書き換わる項目 |
| --- | --- |
| ダッシュボードでコンテキスト長を変える | `"args"` の `--max-context` |
| セットアップをもう一度実行する | `"lib_dirs"`（AMD、ROCm を入れ替えたとき）。`--dspark` / `--no-dspark` を付けたときは `"env"` の `STRATA_DS4_DSPARK` |
| `./setup-deepseek.sh --rewrite-config` | **全部**（手で書き換えた内容は消えます） |
| `./abyss.sh --gpu-tune` | 速かったカーネルの選択を `"env"` に |
| `./abyss.sh --calibrate`（GLM） | `"env"` の `STRATA_GLM_PCIE_SHARE`、`STRATA_GLM_CPU_LANE` |

## そのほかのファイル

| ファイル | 中身 | 書き換えるか |
| --- | --- | --- |
| `Abyss/<設定ファイル名>.shared-settings.json` | チャット画面で決めた、他のアプリにも渡す既定値（`POST /settings`。temperature、無検閲化の入り切りなど） | 画面から変える。消すと既定に戻る |
| `Abyss/.abyss-gpu-tune-<GPU 番号>.json`、`Abyss/abyss-gpu-tune.txt` | `--gpu-tune` の測定結果と、issue に貼る報告 | 触らない |
| `~/.config/project-maya/calibration.json` | `--calibrate` の結果（PC・モデル・コンテキスト・GPU の PCIe リンクとドライバごと。カードを別のスロットに移したりドライバを更新すると古い結果は使われず、1 行で知らせる） | 触らない |
| `Abyss-tools/logs/last-model` | `abyss.sh start` で最後に起動したモデル名 | 触らない |
| `Abyss-tools/abyss.sh`、`common.sh` | 起動・停止の道具。セットアップのたびに最新になる（古いものは `.bak`） | 触らない |

### 各位置の対数確率

`STRATA_GLM_LOGPOS` に未使用の出力パスを指定すると、エンジンの `LOGPOS <カンマ区切りのトークンID>` が各位置の正解トークンと上位トークンの対数確率を記録します。モデルの状態はリセットされます。`STRATA_GLM_LOGPOS_TOPK` は記録する上位トークン数（既定 64）です。`STRATA_GLM_LOGPOS_PREFILL=1` はプレフィルが有効な単一デバイスの DeepSeek V4.1 でプロンプト経路を使用します。既定はトークン経路です。両経路の数値は異なる場合があります。

### DeepSeek V4.1 の画像

DeepSeek V4.1 の画像エンコーダーは、セットアップでは入りません。設定ファイルの `vision` を手で書きます（`exe` は llama.cpp b11870 以降で作った `strata-vision`、`mmproj` はモデルの `deepseek41v` の変換器ファイル、`model` はモデルの最初の GGUF ファイルで、エンコーダーが語彙を読みます）。`STRATA_GLM_VL_BIAS` は、画像の位置で専門家を選ぶときの偏り（`exp_probs_b_vl`。層数 × 専門家数、リトルエンディアンの F32、ヘッダーなし）のファイルを指します。指定がなければ、パックの横の `exp_probs_b_vl.f32` を読みます。これは GGUF に `exp_probs_b_vl.bias` があるとき `tools/iq_pack.py` が書き出します。どちらも無ければモデルは普通に起動し、画像つきの要求は断られます。画像の位置では、この偏りで専門家を選び、Engram は何も足しません。文字の位置は変わりません。HIP のビルドでは画像は使えません。
