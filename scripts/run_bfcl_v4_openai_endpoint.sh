#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-/home/work/.data/lfm2_ko_sft/eval_harnesses}"
BFCL_ROOT="$HARNESS_ROOT/repos/gorilla/berkeley-function-call-leaderboard"
BFCL_VENV="$HARNESS_ROOT/venvs/bfcl"
OUT_ROOT="${OUT_ROOT:-/home/work/.data/lfm2_ko_sft/eval/external_harnesses/bfcl}"
MODEL_NAME="${BFCL_MODEL_NAME:-lfm2-ko-sft-FC}"
TEST_CATEGORY="${TEST_CATEGORY:-simple,parallel,multiple,parallel_multiple,multi_turn_base,multi_turn_miss_param,multi_turn_miss_func}"
LOCAL_SERVER_ENDPOINT="${LOCAL_SERVER_ENDPOINT:-localhost}"
LOCAL_SERVER_PORT="${LOCAL_SERVER_PORT:-1053}"

mkdir -p "$OUT_ROOT"

if [ ! -x "$BFCL_VENV/bin/bfcl" ]; then
  echo "BFCL venv missing. Run scripts/setup_external_eval_harnesses.sh first." >&2
  exit 2
fi

export BFCL_PROJECT_ROOT="$OUT_ROOT"
export LOCAL_SERVER_ENDPOINT
export LOCAL_SERVER_PORT
export REMOTE_OPENAI_BASE_URL="${REMOTE_OPENAI_BASE_URL:-http://${LOCAL_SERVER_ENDPOINT}:${LOCAL_SERVER_PORT}/v1}"
export REMOTE_OPENAI_API_KEY="${REMOTE_OPENAI_API_KEY:-EMPTY}"

cd "$BFCL_ROOT"
"$BFCL_VENV/bin/bfcl" generate \
  --model "$MODEL_NAME" \
  --test-category "$TEST_CATEGORY" \
  --skip-server-setup \
  --num-threads "${NUM_THREADS:-1}"

"$BFCL_VENV/bin/bfcl" evaluate \
  --model "$MODEL_NAME" \
  --test-category "$TEST_CATEGORY"
