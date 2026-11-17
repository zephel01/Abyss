#!/bin/sh
# Abyss の起動・停止・更新をまとめたスクリプト
#   abyss.sh start [model]     必要ならエンジンを再ビルドしてからバックグラウンドで起動（省略時は最後に start したモデル）
#   abyss.sh start -f [model]  このターミナルで起動（Ctrl+C で停止。systemd などの自動起動はこちら）
#   abyss.sh stop              起動中の Abyss を止める
#   abyss.sh restart [model]   stop してから start（省略時は起動中のモデル）
#   abyss.sh status            git の位置、エンジンがソースと一致しているか、起動中か
#   abyss.sh update            git pull してエンジンを再ビルド（起動はしない）
#   abyss.sh build             エンジンの再ビルドだけ（変更分のみ）
#   abyss.sh log [model]       サーバーのログを追いかける (tail -f)
#   abyss.sh models            使えるモデル (run-abyss-*.sh) とポートの一覧
. "$(dirname "$0")/common.sh"
cd "$ABYSS" || exit 1

usage() { sed -n '2,11p' "$TOOLS/abyss.sh" | sed 's/^# \{0,1\}//'; exit 1; }

LAST="$TOOLS/logs/last-model"      # 最後に start したモデル名

# 起動中のモデル名（サーバーの --config .../abyss-<model>.json から）。起動していなければ空
running_model() {
  pgrep -af "$SERVER_PAT" | sed -n 's/.*--config [^ ]*\/abyss-\([^ ]*\)\.json.*/\1/p' | head -n 1
}

# run スクリプトの --port
port_of() { sed -n 's/.*"--port" "\([0-9]*\)".*/\1/p' "$1"; }

# モデル名 ($1, 省略可) から run スクリプトを決めて $script に入れる
# 省略時: run-abyss-*.sh が 1 つならそれ。複数なら 起動中のモデル → 最後に start したモデル
pick_script() {
  if [ -n "$1" ]; then
    script="$ABYSS/run-abyss-$1.sh"
  else
    set -- "$ABYSS"/run-abyss-*.sh
    if [ "$#" -eq 1 ]; then
      script="$1"
    else
      m=$(running_model)
      [ -z "$m" ] && [ -f "$LAST" ] && m=$(cat "$LAST")
      [ -n "$m" ] && [ -x "$ABYSS/run-abyss-$m.sh" ] || { echo "run スクリプトが複数あります。モデル名を指定: $(ls run-abyss-*.sh | sed 's/run-abyss-\(.*\)\.sh/\1/' | tr '\n' ' ')"; exit 1; }
      script="$ABYSS/run-abyss-$m.sh"
    fi
  fi
  [ -x "$script" ] || { echo "$script がありません（Abyss の setup-deepseek.sh で作られる）"; exit 1; }
  model=$(basename "$script" .sh | sed 's/^run-abyss-//')
  cfg="$ABYSS/abyss-$model.json"
  log="$TOOLS/logs/server-$model.log"
}

# エンジンの再ビルド（変更分だけ）。build/ がなければ README の手順で構成してから
build_engine() {
  [ -f "$ABYSS/build/build.ninja" ] || { echo "!! $ABYSS/build がありません（Abyss の setup-deepseek.sh を実行）"; return 1; }
  if [ -z "$(ninja_dry)" ]; then echo "== エンジンは最新"; return 0; fi
  echo "== エンジンを再ビルド（変更分のみ）"
  "$CMAKE" --build "$ABYSS/build" --target strata -j "$(nproc)" | grep -E "error|Linking CXX executable" || true
  [ -z "$(ninja_dry)" ]
}
# まだビルドしていない変更があれば、その作業の一覧（なければ空）
ninja_dry() { "$ABYSS/.venv/bin/ninja" -C "$ABYSS/build" -n strata 2>/dev/null | grep -v "^ninja: no work" | grep -v "^ninja: Entering"; }

cmd_start() {
  fg=
  [ "$1" = "-f" ] && { fg=1; shift; }
  pick_script "$1"
  if running; then echo "Abyss はもう起動しています"; exit 1; fi

  build_engine || echo "!! ビルドに失敗。今のエンジンで起動します"
  mkdir -p "$TOOLS/logs"
  echo "$model" > "$LAST"
  echo "== モデル: $model  ポート: $(port_of "$script")"
  [ -n "$fg" ] && exec "$script"

  # バックグラウンド: 端末から切り離して起動し、ready が出るまで待つ（Ctrl+C で待つのをやめてもサーバーは動き続ける）
  : > "$log"
  setsid nohup "$script" >"$log" 2>&1 </dev/null &
  pid=$!
  echo "== 起動中 (pid $pid)  ログ: $log"
  echo "   エンジンのログ: $ABYSS/abyss-$model.log"
  echo "   読み込みに 1 分ほどかかる（密な重みを GPU に載せ、全エキスパートを VRAM と RAM に読み込む）"
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q '^ready:' "$log"; then
      grep -A1 '^ready:' "$log"
      echo "== 起動しました。止めるときは $TOOLS/abyss.sh stop"
      exit 0
    fi
    sleep 2
  done
  echo "!! 起動に失敗しました。ログの最後:"
  tail -n 20 "$log"
  exit 1
}

# $1 秒待って、止まったら 0
wait_stop() {
  i=0
  while running; do
    [ "$i" -ge "$1" ] && return 1
    sleep 1; i=$((i + 1))
  done
}

# SIGTERM をサーバーにだけ送る。server.py は SIGTERM を Ctrl+C と同じに扱い、エンジンに QUIT を送って片付けてから終わる
# （SIGINT は使わない: setsid nohup ... & で起動したサーバーは SIGINT を無視したまま引き継ぐ。Maya-tools と同じ）
# 処理中のリクエストは最大 40 秒で答えを閉じ (STRATA_ENGINE_WRAP_S)、エンジンの終了を最大 50 秒待つ (STRATA_ENGINE_QUIT_S)
# ので 100 秒待つ。止まらなければエンジンにも SIGTERM → SIGKILL
cmd_stop() {
  if ! running; then echo "Abyss は起動していません"; return 0; fi
  procs | sed 's/^/  /'
  echo "== 停止中 (サーバーに SIGTERM)"
  pkill -TERM -f "$SERVER_PAT"
  wait_stop 100 && { echo "== 停止しました"; return 0; }
  echo "== 止まらないのでエンジンにも SIGTERM"
  pkill -TERM -f "$PAT"
  wait_stop 10 && { echo "== 停止しました"; return 0; }
  echo "== 強制終了 (SIGKILL)"
  pkill -KILL -f "$PAT"
  wait_stop 5 && { echo "== 停止しました"; return 0; }
  echo "止められませんでした:"; procs | sed 's/^/  /'
  exit 1
}

cmd_status() {
  echo "git        : $(git log -1 --format='%h %s' | cut -c1-90)"
  git fetch -q 2>/dev/null && echo "未取得     : $(git rev-list --count HEAD..@{u} 2>/dev/null || echo '?') commits (origin)"
  eng="$ABYSS/build/abyss"
  [ -x "$eng" ] || eng="$ABYSS/build/strata"   # (v1.3.163 まではこの名前)
  if [ -x "$eng" ]; then
    arch=$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[A-Z]*=//p' "$ABYSS/build/CMakeCache.txt" 2>/dev/null)
    echo "engine     : $eng  [arch $arch, $(date -r "$eng" '+%Y-%m-%d %H:%M')]"
    if [ -z "$(ninja_dry)" ]; then echo "src        : 一致 (最新)"; else echo "src        : ソースの方が新しい -> abyss.sh build"; fi
  else
    echo "engine     : 未ビルド（Abyss の setup-deepseek.sh を実行）"
  fi
  if running; then echo "server     : 起動中 ($(running_model))"; procs | sed 's/^/             /'; else echo "server     : 停止中"; fi
}

# run-abyss-*.sh の一覧（* = 起動中、> = 省略時に選ばれるモデル）
cmd_models() {
  cur=$(running_model)
  last=$(cat "$LAST" 2>/dev/null)
  for f in "$ABYSS"/run-abyss-*.sh; do
    [ -x "$f" ] || continue
    m=$(basename "$f" .sh | sed 's/^run-abyss-//')
    mark=" "
    [ "$m" = "$last" ] && mark=">"
    [ "$m" = "$cur" ] && mark="*"
    printf '%s %-22s ポート %s\n' "$mark" "$m" "$(port_of "$f")"
  done
  echo "(* 起動中  > 最後に start したモデル。start / restart でモデル名を省くとこれを使う)"
}

need_stopped() {
  if running; then
    echo "Abyss が起動中です。止めてから実行してください:"
    procs | sed 's/^/  /'
    exit 1
  fi
}

cmd_update() {
  need_stopped
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Abyss に未コミットの変更があります。コミットか git stash で退避してから実行してください:"
    git status --short --untracked-files=no
    exit 1
  fi
  old=$(git rev-parse HEAD)
  git pull -q --ff-only || { echo "fast-forward できません。手動で rebase / reset してください"; exit 1; }
  new=$(git rev-parse HEAD)
  if [ "$old" = "$new" ]; then
    echo "== 更新なし"
  else
    echo "== 取り込んだコミット"
    git log --oneline "$old..$new" | cut -c1-120
    if git diff "$old" "$new" -- setup.py | grep -q '^[-+]LLAMA_CPP_COMMIT'; then
      echo "!! llama.cpp の固定コミットが変わった: rm -rf third_party/llama.cpp してから Abyss の setup-deepseek.sh を実行（取り直す）"
    fi
    if git diff "$old" "$new" -- tools/iq_pack.py tools/strata_tokenizer.py | grep -q '^[-+]'; then
      echo "!! pack を作るツールが変わった: 必要なら pack を消して Abyss の setup-deepseek.sh で作り直す"
    fi
  fi
  build_engine
}

cmd_log() {
  pick_script "$1"
  [ -f "$log" ] || { echo "$log がありません（start でバックグラウンド起動したときに作られる）"; exit 1; }
  exec tail -n 50 -f "$log"
}

cmd="$1"
[ -n "$cmd" ] && shift
case "$cmd" in
  start)     cmd_start "$@" ;;
  stop)      cmd_stop ;;
  restart)   [ -z "$1" ] && set -- $(running_model); cmd_stop && cmd_start "$@" ;;
  status)    cmd_status ;;
  update)    cmd_update ;;
  build)     need_stopped; build_engine ;;
  log)       cmd_log "$@" ;;
  models)    cmd_models ;;
  *)         usage ;;
esac
