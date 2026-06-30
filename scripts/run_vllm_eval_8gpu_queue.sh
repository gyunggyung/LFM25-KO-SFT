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

if [ -n "${GPU_IDS:-}" ]; then
  IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
else
  gpu_count="${GPU_COUNT:-8}"
  GPU_ID_LIST=()
  for ((i = 0; i < gpu_count; i++)); do
    GPU_ID_LIST+=("$i")
  done
fi
gpu_count="${#GPU_ID_LIST[@]}"
if [ "$gpu_count" -eq 0 ]; then
  echo "No GPUs configured. Set GPU_COUNT or GPU_IDS." >&2
  exit 2
fi
slot=0
pids=()
labels=()
failed=0

drop_pid() {
  local target="$1"
  local next_pids=()
  local next_labels=()
  local i
  for i in "${!pids[@]}"; do
    if [ "${pids[$i]}" != "$target" ]; then
      next_pids+=("${pids[$i]}")
      next_labels+=("${labels[$i]}")
    fi
  done
  pids=("${next_pids[@]}")
  labels=("${next_labels[@]}")
}

label_for_pid() {
  local target="$1"
  local i
  for i in "${!pids[@]}"; do
    if [ "${pids[$i]}" = "$target" ]; then
      echo "${labels[$i]}"
      return
    fi
  done
}

wait_for_one() {
  local done_pid="" status=0 done_label=""
  set +e
  wait -n -p done_pid
  status=$?
  set -e
  if [ -n "$done_pid" ]; then
    done_label="$(label_for_pid "$done_pid")"
    drop_pid "$done_pid"
    if [ "$status" -eq 0 ]; then
      echo "finished status=0 pid=$done_pid label=$done_label"
    else
      failed=$((failed + 1))
      echo "finished status=$status pid=$done_pid label=$done_label" >&2
    fi
  fi
}

launch_one() {
  local model="$1"
  local task_group="$2"
  local gpu="$3"
  local label task_label log_path
  if [ "$(basename "$model")" = "final_full" ]; then
    label="$(basename "$(dirname "$model")")_final_full"
  else
    label="$(basename "$model")"
  fi
  label="$(echo "$label" | tr '/:' '__' | tr -cd '[:alnum:]_.-' | cut -c1-120)"
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
  labels+=("${label}.${task_label}.gpu${gpu}")
  echo "launched gpu=$gpu model=$model tasks=$task_group log=$log_path"
}

for model in "${MODELS[@]}"; do
  for task_group in "${TASK_GROUPS[@]}"; do
    gpu="${GPU_ID_LIST[$((slot % gpu_count))]}"
    launch_one "$model" "$task_group" "$gpu"
    slot=$((slot + 1))
    if [ "${#pids[@]}" -ge "$gpu_count" ]; then
      wait_for_one
    fi
  done
done

while [ "${#pids[@]}" -gt 0 ]; do
  wait_for_one
done

"$WORK_DIR/scripts/summarize_lm_eval_results.py" "$OUT_ROOT/$RUN_ID" > "$OUT_ROOT/$RUN_ID/SUMMARY.md"
echo "summary=$OUT_ROOT/$RUN_ID/SUMMARY.md"
echo "failed_jobs=$failed"
if [ "${FAIL_ON_EVAL_ERROR:-0}" = "1" ] && [ "$failed" -gt 0 ]; then
  exit 1
fi
