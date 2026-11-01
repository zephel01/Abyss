# Config files

[日本語](SETTINGS.ja.md)

Everything about Abyss's config files is on this page, written for someone using Abyss for the first time:
1. which files there are and where they are
2. where to change what you want to change
3. what to do after a change so that it takes effect

## Contents

- [Setup and config](#setup-and-config)
- [What to change where](#what-to-change-where)
- [Where the files are](#where-the-files-are)
- [Finding the config in use](#finding-the-config-in-use)
- [Changing a config](#changing-a-config)
- [A config file (example)](#a-config-file-example)
- [Every key](#every-key)
- [Keys Abyss writes itself](#keys-abyss-writes-itself)
- [Other files](#other-files)

## Setup and config

| | Setup | Config |
| --- | --- | --- |
| When | once at the start (again when you add a model) | afterwards, whenever you need to |
| What | checks the PC, builds the engine, downloads the model, makes the pack, **writes the config file and the start script** | open the config file it wrote, change it, restart the server |
| Where | [MANUAL: Setup in one command](MANUAL.md#setup-in-one-command) | this page |

The last thing setup writes is the "config file" this page is about. Running setup again keeps an existing config file
as it is (it writes it again only with `--rewrite-config`).

## What to change where

"The config file" here is `abyss-<name>.json` in the Abyss folder.

| To change | Where |
| --- | --- |
| the context length | the number after `"--max-context"` in the config's `"args"`. Changing it on the dashboard saves it there too |
| how random the answers are (temperature, ...) | the config's `"sampling"` |
| the default thinking depth | the config's `"reasoning_effort"` |
| the port | the config's `"port"` **and** `"--port"` in the start script `run-abyss-<name>.sh`, both |
| connecting from other PCs | `"host": "0.0.0.0"` and `"api_key": "<secret>"` in the config (always both) |
| which GPU | the config's `"gpu"` (AMD: `"gpu_pci"` too) |
| uncensoring | the config's `"uncensor"` ([UNCENSOR](UNCENSOR.md)) |
| fine engine tuning (how RAM is used, ...) | the config's `"env"` (the variables: [MANUAL: Settings](MANUAL.md#settings)) |
| the model name the API reports | the config's `"model_name"` |

Every change needs a **server restart** ([Changing a config](#changing-a-config)).

## Where the files are

With the clone in `~/llm/apps/Abyss`. Setup makes `Abyss-data` and `Abyss-tools` beside it.

```
~/llm/apps/
├── Abyss/                                  the program (updated with git)
│   ├── abyss-dsv4-q2_k_xl.json             * a config file (written by setup-deepseek.sh)
│   ├── run-abyss-dsv4-q2_k_xl.sh             its start script
│   ├── abyss-dsv4-q2_k_xl.log                the engine's log
│   ├── abyss-dsv4-q2_k_xl.shared-settings.json   the Chat page's shared settings (if any)
│   ├── maya-<model>.json                   * a config file (GLM-5.3-Flash, written by ./abyss.sh)
│   ├── run-maya-<model>.sh                   its start script
│   └── build/strata                        the engine (AMD: build-hip/strata)
├── Abyss-data/models/...                   the model files and packs (tens to hundreds of GB)
└── Abyss-tools/
    ├── abyss.sh                            start | stop | restart | status | log | models
    └── logs/
        ├── server-dsv4-q2_k_xl.log         the server's log (abyss.sh log shows it)
        └── last-model                      the model started last
```

Setup picks the `<name>`:

| Setup | Config file | Start script | Default port |
| --- | --- | --- | --- |
| `./setup-deepseek.sh` (DeepSeek V4 / V4.1, ...) | `abyss-<name>.json`; the name defaults to `dsv4-<quant>` (e.g. `abyss-dsv4-q2_k_xl.json`), `--name` changes it | `run-abyss-<name>.sh` | 8082 |
| `./abyss.sh` (GLM-5.3-Flash) | `maya-<model>.json`; on AMD `maya-<model>-hip.json` | `run-maya-<model>.sh` (Windows: `.bat`) | 8080 |
| by hand ([MANUAL: Configure and start](MANUAL.md#configure-and-start)) | any name, in the Abyss folder | none (`serve/server.py --config <file>`) | your choice |

With several models installed, each has its own config file and start script.

## Finding the config in use

A start script and its config share the name: `run-abyss-X.sh` uses `abyss-X.json`.

```sh
../Abyss-tools/abyss.sh models
```

prints, for example:

```
* dsv4-q2_k_xl           ポート 8082
> dsv41-iq2xxs           ポート 8082
(* 起動中  > 最後に start したモデル。start / restart でモデル名を省くとこれを使う)
```

`*` is the running model, `>` the one started last. Here `dsv4-q2_k_xl` runs, so its config is
`~/llm/apps/Abyss/abyss-dsv4-q2_k_xl.json`.

To be sure, look in the start script: the file after `--config` is its config.

```sh
grep -o -- '--config" "[^"]*' ~/llm/apps/Abyss/run-abyss-dsv4-q2_k_xl.sh
```

## Changing a config

1. **Make a backup**: `cp abyss-dsv4-q2_k_xl.json abyss-dsv4-q2_k_xl.json.bak`
2. **Open it in a text editor and change it.** It is JSON:
   - strings in `"` (numbers and `true` / `false` without)
   - a `,` between entries, none after the last one
   - in `"args"`, an option and its value are separate strings (`"--max-context", "65536"`)
3. **Restart the server**: `../Abyss-tools/abyss.sh restart` (GLM: stop it, then `./run-maya-<model>.sh`)
4. **Check the log**: `../Abyss-tools/abyss.sh log` shows the server's log (`Abyss-tools/logs/server-<name>.log`);
   a JSON mistake or the reason a start failed is there. The engine's detailed log is the file in the config's
   `"log"` (`Abyss/abyss-<name>.log`).

If it stops working, put the backup from step 1 back.

## A config file (example)

As `./setup-deepseek.sh` writes it (an AMD GPU, the DSpark drafter on):

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

## Every key

**The usual ones**

| Key | Meaning | Example |
| --- | --- | --- |
| `exe` | the engine | `".../Abyss/build/strata"` (AMD: `build-hip/strata`) |
| `args` | the engine's options: `--glm-pack` the pack folder, `--max-context` the context in tokens, `--prefill auto` how much of a prompt is read at once | the example above |
| `cwd` | the folder the engine runs in | the Abyss folder |
| `tokenizer` | the pack's `tokenizer` folder (it holds the chat template) | `".../pack/tokenizer"` |
| `model_name` | the model name the API reports | `"deepseek-v4-flash"` |
| `port` | the server's port; keep the start script's `--port` the same | `8082` |
| `log` | where the engine writes its log | `".../abyss-<name>.log"` |
| `sampling` | defaults for requests that leave them out: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, `frequency_penalty`, `penalty_last_n`, `seed`. A request's own value wins | `{"temperature": 1.0, "top_p": 0.95}` |
| `reasoning_effort` | the default thinking depth (`none`, `low`, `high`, `max`) | `"high"` |
| `thinking_budget` | the most tokens a thinking block may take (default 32768; 0 = no cap) | `16384` |
| `env` | fine engine settings (environment variables); the list is in [MANUAL: Settings](MANUAL.md#settings) | `{"STRATA_GLM_RAM_GB": "60"}` |
| `uncensor` | an uncensoring file ([UNCENSOR](UNCENSOR.md)) | `".../dsv4.abl"` |

**GPUs**

| Key | Meaning | Example |
| --- | --- | --- |
| `gpu` | the GPU number(s); several split the layers (the first is the main card) | `[0]`, `[0, 1]` |
| `layer_split` | with several GPUs, the first layer of each later GPU's share; default `"auto"` | `"18"`, `"16,32"` |
| `backend` | `"hip"` for AMD; left out for NVIDIA | `"hip"` |
| `gpu_pci` | AMD: the GPUs' PCI addresses, so the same cards are used when a reboot renumbers them (setup writes it) | `["0000:c7:00.0"]` |
| `lib_dirs` | AMD: where the ROCm libraries are (setup writes it) | `["/opt/rocm/lib"]` |

**Access and security**

| Key | Meaning | Example |
| --- | --- | --- |
| `host` | the address the server listens on; default `127.0.0.1` (this PC only), `0.0.0.0` lets other PCs in | `"0.0.0.0"` |
| `api_key` | the key `/v1/*` requires (`Authorization: Bearer ...` or `x-api-key`), several separated by commas. Always set it when `host` is open | `"my-secret"` |
| `allow_shutdown` | `false`: `POST /api/shutdown` (stopping the server and engine from another program, e.g. when the engine runs away) is refused. Default on: allowed with the `api_key`, or from this PC when there is no key. Over Tailscale, set an `api_key` ([MANUAL](MANUAL.md#using-it)) | `false` |
| `fit_max_tokens` | `true`: a `max_tokens` beyond the context left is shortened instead of refused with a 400 | `true` |
| `mcp_servers` | MCP servers the Chat page can use (Claude Desktop's format) | `{"name": {...}}` |
| `installer` | what setup chose (the model and quant, for the Monitor's label); setup writes it, nothing reads it to decide | `{"quant": "UD-IQ2_XXS"}` |

**Images (GLM-5.3-Flash by setup; DeepSeek V4.1 by hand, see below)**: `vision` holds the image encoder's settings (`exe`, `mmproj`, `model`, `gpu`,
`resident`, `threads`, `max_tokens`, `no_flash_attn`). `./abyss.sh` writes it; it is normally left alone.

Unknown keys are ignored. A bad value in `sampling`, though, stops the server from starting (so a typo cannot quietly
change how it answers).

## Keys Abyss writes itself

These change part of the config file; the other keys stay as they are.

| Action | What changes |
| --- | --- |
| changing the context length on the dashboard | `--max-context` in `"args"` |
| running setup again | `"lib_dirs"` (AMD, after a ROCm switch); with `--dspark` / `--no-dspark`, `STRATA_DS4_DSPARK` in `"env"` |
| `./setup-deepseek.sh --rewrite-config` | **everything** (your own changes are lost) |
| `./abyss.sh --gpu-tune` | the faster kernel choices, into `"env"` |
| `./abyss.sh --calibrate` (GLM) | `STRATA_GLM_PCIE_SHARE`, `STRATA_GLM_CPU_LANE` in `"env"` |

## Other files

| File | What it holds | Edit it? |
| --- | --- | --- |
| `Abyss/<config name>.shared-settings.json` | the defaults chosen on the Chat page that other apps get too (`POST /settings`: temperature, the uncensoring switch, ...) | change them on the page; deleting the file restores the defaults |
| `Abyss/.abyss-gpu-tune-<GPU number>.json`, `Abyss/abyss-gpu-tune.txt` | `--gpu-tune`'s measurements and the report for an issue | no |
| `~/.config/project-maya/calibration.json` | `--calibrate`'s results (per PC, model, context, and the GPU's PCIe link and driver: a card moved to another slot or a driver update leaves the old result unused, with one line saying so) | no |
| `Abyss-tools/logs/last-model` | the model `abyss.sh start` started last | no |
| `Abyss-tools/abyss.sh`, `common.sh` | the start / stop tools; setup brings them up to date (the old ones kept as `.bak`) | no |

### Per-position log probabilities

Set `STRATA_GLM_LOGPOS` to a new output path to enable the engine command `LOGPOS <comma-separated-token-ids>`. It writes the target and top-k log probabilities for each next-token position as tab-separated rows and resets the model state. `STRATA_GLM_LOGPOS_TOPK` sets the number of top tokens (default 64). `STRATA_GLM_LOGPOS_PREFILL=1` uses the prompt path on a single-device DeepSeek V4.1 with prefill enabled; the default is the token path. The paths can differ numerically. Each tab-separated row contains position, target ID, target log probability, top ID, top log probability, hit (0 or 1), then top-k `ID:log_probability` fields. Multiple requests append to the same file and restart positions at zero; wait for `LPDONE` before sending `QUIT`.

### Images on DeepSeek V4.1

Setup does not install the image encoder for DeepSeek V4.1; add the `vision` entry to the config by hand (`exe` is a `strata-vision` built from llama.cpp b11870 or later, `mmproj` is the model's `deepseek41v` projector file, `model` is the first GGUF file of the model, which the encoder reads the vocabulary from). `STRATA_GLM_VL_BIAS` names the file with the router bias for image positions (`exp_probs_b_vl`, layers x experts, little-endian F32, no header). Without it Abyss reads `exp_probs_b_vl.f32` next to the pack, which `tools/iq_pack.py` writes when the GGUF has `exp_probs_b_vl.bias`. If neither exists the model still starts, and a request with an image is refused. At image positions the router uses this bias and the Engram adds nothing; text positions are unchanged. HIP builds have no images.
