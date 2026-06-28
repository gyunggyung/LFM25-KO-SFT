#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-/home/work/.data/lfm2_ko_sft/eval_harnesses}"
IFBENCH_ROOT="$HARNESS_ROOT/repos/IFBench"
IFBENCH_VENV="$HARNESS_ROOT/venvs/ifbench"
PORT="${PORT:-1053}"
MODEL_NAME="${MODEL_NAME:-lfm2-ko-sft}"
API_BASE="${API_BASE:-http://localhost:${PORT}/v1}"
API_KEY="${API_KEY:-EMPTY}"
INPUT_FILE="${INPUT_FILE:-$IFBENCH_ROOT/data/IFBench_test.jsonl}"
OUT_ROOT="${OUT_ROOT:-/home/work/.data/lfm2_ko_sft/eval/external_harnesses/ifbench}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_ifbench}"
WORKERS="${WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"

if [ ! -x "$IFBENCH_VENV/bin/python" ]; then
  echo "IFBench venv missing. Run scripts/setup_external_eval_harnesses.sh first." >&2
  exit 2
fi
if [ ! -f "$INPUT_FILE" ]; then
  echo "IFBench input missing: $INPUT_FILE" >&2
  exit 2
fi

RUN_DIR="$OUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

cd "$IFBENCH_ROOT"
"$IFBENCH_VENV/bin/python" generate_responses.py \
  --api-base "$API_BASE" \
  --model "$MODEL_NAME" \
  --input-file "$INPUT_FILE" \
  --output-file "$RUN_DIR/${MODEL_NAME}-responses.jsonl" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --api-key "$API_KEY" \
  --workers "$WORKERS" \
  --seed 42 \
  --resume

"$IFBENCH_VENV/bin/python" -m run_eval \
  --input_data "$INPUT_FILE" \
  --input_response_data "$RUN_DIR/${MODEL_NAME}-responses.jsonl" \
  --output_dir "$RUN_DIR/eval"

echo "ifbench_run_dir=$RUN_DIR"
