#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_vllm_queue}"
OUT_ROOT="${OUT_ROOT:-/home/work/.data/lfm2_ko_sft/eval}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"

MODELS_FILE="${MODELS_FILE:-$WORK_DIR/configs/eval_models_20260628.txt}"
TASK_GROUPS_FILE="${TASK_GROUPS_FILE:-$WORK_DIR/configs/eval_task_groups_20260628.txt}"

if [ ! -f "$MODELS_FILE" ]; then
  echo "Missing MODELS_FILE: $MODELS_FILE" >&2
  exit 2
fi
if [ ! -f "$TASK_GROUPS_FILE" ]; then
  echo "Missing TASK_GROUPS_FILE: $TASK_GROUPS_FILE" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/$RUN_ID/logs"

mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODELS_FILE")
mapfile -t TASK_GROUPS < <(grep -vE '^\s*(#|$)' "$TASK_GROUPS_FILE")

gpu_count="${GPU_COUNT:-8}"
slot=0
pids=()

launch_one() {
  local model="$1"
  local task_group="$2"
  local gpu="$3"
  local label task_label log_path
  label="$(basename "$model" | tr '/:' '__')"
  task_label="$(echo "$task_group" | tr ',' '_' | tr -cd '[:alnum:]_-' | cut -c1-80)"
  log_path="$OUT_ROOT/$RUN_ID/logs/${label}.${task_label}.gpu${gpu}.log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    MODEL_ID="$model" \
    MODEL_LABEL="${label}.${task_label}" \
    TASKS="$task_group" \
    RUN_ID="$RUN_ID" \
    OUT_ROOT="$OUT_ROOT" \
    VLLM_ENV="$VLLM_ENV" \
    TENSOR_PARALLEL_SIZE=1 \
    bash "$WORK_DIR/scripts/run_vllm_lm_eval_matrix.sh"
  ) >"$log_path" 2>&1 &
  pids+=("$!")
  echo "launched gpu=$gpu model=$model tasks=$task_group log=$log_path"
}

for model in "${MODELS[@]}"; do
  for task_group in "${TASK_GROUPS[@]}"; do
    gpu=$((slot % gpu_count))
    launch_one "$model" "$task_group" "$gpu"
    slot=$((slot + 1))
    if [ "${#pids[@]}" -ge "$gpu_count" ]; then
      wait -n
      alive=()
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          alive+=("$pid")
        fi
      done
      pids=("${alive[@]}")
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$WORK_DIR/scripts/summarize_lm_eval_results.py" "$OUT_ROOT/$RUN_ID" > "$OUT_ROOT/$RUN_ID/SUMMARY.md"
echo "summary=$OUT_ROOT/$RUN_ID/SUMMARY.md"
