# 各スクリプト共通: Abyss の場所と、動いているかの判定
ABYSS="${ABYSS_DIR:-@ABYSS@}"      # (setup-deepseek.sh が Abyss のフォルダをここに書く)
PY="$ABYSS/.venv/bin/python"
CMAKE="$ABYSS/.venv/bin/cmake"
TOOLS="$(cd "$(dirname "$0")" && pwd)"

# サーバー (python <Abyss>/serve/server.py) かエンジン本体 (<Abyss>/build/abyss、旧名 build/strata)。コマンドの先頭で判定する
# Strata / Maya も同じ名前 (serve/server.py, strata) なので、Abyss のフォルダのパスで区別する
ABYSS_RE=$(printf '%s' "$ABYSS" | sed 's/[.[\*^$]/\\&/g')
SERVER_PAT="^[^ ]*python[0-9.]* $ABYSS_RE/serve/server\.py "
PAT="$SERVER_PAT|^$ABYSS_RE/build/(abyss|strata)( |\$)"
procs() { pgrep -af "$PAT"; }
running() { pgrep -f "$PAT" >/dev/null; }
