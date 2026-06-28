#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
HARNESS_ROOT="${HARNESS_ROOT:-/home/work/.data/lfm2_ko_sft/eval_harnesses}"

mkdir -p "$HARNESS_ROOT/repos" "$HARNESS_ROOT/venvs" "$WORK_DIR/logs/eval"

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    git -C "$dir" fetch --depth 1 origin main || git -C "$dir" fetch origin
    git -C "$dir" pull --ff-only || true
  else
    git clone --depth 1 "$url" "$dir"
  fi
}

make_venv() {
  local name="$1"
  local py="${PYTHON_BIN:-python3}"
  local venv="$HARNESS_ROOT/venvs/$name"
  if [ ! -x "$venv/bin/python" ]; then
    "$py" -m venv "$venv"
  fi
  "$venv/bin/python" -m pip install -U pip wheel setuptools
}

echo "harness_root=$HARNESS_ROOT"

clone_or_update https://github.com/ShishirPatil/gorilla.git "$HARNESS_ROOT/repos/gorilla"
clone_or_update https://github.com/sierra-research/tau2-bench.git "$HARNESS_ROOT/repos/tau2-bench"
clone_or_update https://github.com/allenai/IFBench.git "$HARNESS_ROOT/repos/IFBench"
clone_or_update https://github.com/facebookresearch/Multi-IF.git "$HARNESS_ROOT/repos/Multi-IF"
clone_or_update https://github.com/huggingface/lighteval.git "$HARNESS_ROOT/repos/lighteval"

mkdir -p "$HARNESS_ROOT/repos/Multi-IF/data"
clone_or_update https://huggingface.co/datasets/facebook/Multi-IF "$HARNESS_ROOT/repos/Multi-IF/data/Multi-IF"

make_venv bfcl
"$HARNESS_ROOT/venvs/bfcl/bin/python" -m pip install -e "$HARNESS_ROOT/repos/gorilla/berkeley-function-call-leaderboard"

make_venv ifbench
"$HARNESS_ROOT/venvs/ifbench/bin/python" -m pip install -r "$HARNESS_ROOT/repos/IFBench/requirements.txt" openai httpx tqdm

make_venv multi-if
"$HARNESS_ROOT/venvs/multi-if/bin/python" -m pip install -r "$HARNESS_ROOT/repos/Multi-IF/requirements.txt"

make_venv aa-omniscience
"$HARNESS_ROOT/venvs/aa-omniscience/bin/python" -m pip install -U "lighteval[accelerate]" inspect-ai datasets openai

if command -v uv >/dev/null 2>&1; then
  (
    cd "$HARNESS_ROOT/repos/tau2-bench"
    uv sync
  )
else
  echo "uv not found; tau2-bench repo cloned but uv sync not run" >&2
fi

cat > "$HARNESS_ROOT/README.status.txt" <<EOF
External harness setup completed at $(TZ=Asia/Seoul date '+%F %T KST')

Repos:
- BFCL: $HARNESS_ROOT/repos/gorilla/berkeley-function-call-leaderboard
- tau2-bench: $HARNESS_ROOT/repos/tau2-bench
- IFBench: $HARNESS_ROOT/repos/IFBench
- Multi-IF: $HARNESS_ROOT/repos/Multi-IF
- Multi-IF data: $HARNESS_ROOT/repos/Multi-IF/data/Multi-IF
- lighteval: $HARNESS_ROOT/repos/lighteval

Venvs:
- $HARNESS_ROOT/venvs/bfcl
- $HARNESS_ROOT/venvs/ifbench
- $HARNESS_ROOT/venvs/multi-if
- $HARNESS_ROOT/venvs/aa-omniscience
EOF

cat "$HARNESS_ROOT/README.status.txt"
