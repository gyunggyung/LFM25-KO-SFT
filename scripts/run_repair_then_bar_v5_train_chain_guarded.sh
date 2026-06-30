#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_TRAIN:-0}" != "1" ]; then
  cat <<'MSG'
Refusing to start training.

This chain uses GPUs. Run only after explicit approval:
  ALLOW_TRAIN=1 bash scripts/run_repair_then_bar_v5_train_chain_guarded.sh
MSG
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs/train_chain/$(date -u +%Y%m%dT%H%M%SZ)_repair_then_bar_v5}"

REPAIR_OUTPUT_DIR="${REPAIR_OUTPUT_DIR:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630}"
REPAIR_HUB_MODEL_ID="${REPAIR_HUB_MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT}"
BAR_V5_OUTPUT_DIR="${BAR_V5_OUTPUT_DIR:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT-20260630}"
BAR_V5_HUB_MODEL_ID="${BAR_V5_HUB_MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT}"

UPLOAD_TO_HUB="${UPLOAD_TO_HUB:-1}"
WAIT_FOR_UPLOADS="${WAIT_FOR_UPLOADS:-1}"
REPAIR_MASTER_PORT="${REPAIR_MASTER_PORT:-29631}"
BAR_V5_MASTER_PORT="${BAR_V5_MASTER_PORT:-29641}"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"

upload_background() {
  local folder="$1"
  local repo_id="$2"
  local message="$3"
  local label="$4"
  local log_path="$LOG_DIR/${label}.upload.log"
  local pid_path="$LOG_DIR/${label}.upload.pid"

  if [ "$UPLOAD_TO_HUB" != "1" ]; then
    echo "skip_upload label=$label repo=$repo_id folder=$folder"
    return 0
  fi

  if [ ! -d "$folder" ]; then
    echo "missing upload folder: $folder" >&2
    return 1
  fi

  "$VENV/bin/python" "$WORK_DIR/scripts/upload_model_folder_to_hf.py" \
    --folder "$folder" \
    --repo-id "$repo_id" \
    --commit-message "$message" >"$log_path" 2>&1 &
  echo "$!" >"$pid_path"
  echo "upload_started label=$label pid=$(cat "$pid_path") repo=$repo_id log=$log_path"
}

wait_uploads() {
  local failed=0
  local pid_file pid label
  shopt -s nullglob
  for pid_file in "$LOG_DIR"/*.upload.pid; do
    pid="$(cat "$pid_file")"
    label="$(basename "$pid_file" .upload.pid)"
    if wait "$pid"; then
      echo "upload_finished label=$label status=0"
    else
      failed=$((failed + 1))
      echo "upload_finished label=$label status=failed" >&2
    fi
  done
  shopt -u nullglob
  if [ "$failed" -gt 0 ]; then
    return 1
  fi
}

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "log_dir=$LOG_DIR"
echo "repair_output_dir=$REPAIR_OUTPUT_DIR"
echo "repair_hub_model_id=$REPAIR_HUB_MODEL_ID"
echo "bar_v5_output_dir=$BAR_V5_OUTPUT_DIR"
echo "bar_v5_hub_model_id=$BAR_V5_HUB_MODEL_ID"
echo "upload_to_hub=$UPLOAD_TO_HUB"

echo "stage=repair_sft start_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
env \
  ALLOW_TRAIN=1 \
  PUSH_TO_HUB= \
  RUN_ID="${REPAIR_RUN_ID:-ko-cpt-repair-sft-20260630}" \
  STAGE_NAME="${REPAIR_STAGE_NAME:-ko_cpt_repair_sft}" \
  OUTPUT_DIR="$REPAIR_OUTPUT_DIR" \
  HUB_MODEL_ID="$REPAIR_HUB_MODEL_ID" \
  MASTER_PORT="$REPAIR_MASTER_PORT" \
  MAX_SEQ_LENGTH="${REPAIR_MAX_SEQ_LENGTH:-4096}" \
  PER_DEVICE_TRAIN_BATCH_SIZE="${REPAIR_PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  GRADIENT_ACCUMULATION_STEPS="${REPAIR_GRADIENT_ACCUMULATION_STEPS:-8}" \
  LEARNING_RATE="${REPAIR_LEARNING_RATE:-1e-6}" \
  NUM_TRAIN_EPOCHS="${REPAIR_NUM_TRAIN_EPOCHS:-1}" \
  SAVE_STEPS="${REPAIR_SAVE_STEPS:-500}" \
  SAVE_TOTAL_LIMIT="${REPAIR_SAVE_TOTAL_LIMIT:-2}" \
  bash scripts/run_cpt_repair_sft_train_guarded.sh 2>&1 | tee "$LOG_DIR/repair_sft.train.log"

if [ ! -d "$REPAIR_OUTPUT_DIR/final_full" ]; then
  echo "repair final checkpoint missing: $REPAIR_OUTPUT_DIR/final_full" >&2
  exit 1
fi

upload_background \
  "$REPAIR_OUTPUT_DIR/final_full" \
  "$REPAIR_HUB_MODEL_ID" \
  "Upload KO-CPT repair SFT final_full" \
  "repair_sft"

echo "stage=bar_exam_v5_sft start_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
env \
  ALLOW_TRAIN=1 \
  PUSH_TO_HUB= \
  MODEL_PATH="$REPAIR_OUTPUT_DIR/final_full" \
  RUN_ID="${BAR_V5_RUN_ID:-ko-cpt-repair-bar-exam-v5-sft-20260630}" \
  STAGE_NAME="${BAR_V5_STAGE_NAME:-ko_cpt_repair_bar_exam_v5_sft}" \
  OUTPUT_DIR="$BAR_V5_OUTPUT_DIR" \
  HUB_MODEL_ID="$BAR_V5_HUB_MODEL_ID" \
  MASTER_PORT="$BAR_V5_MASTER_PORT" \
  MAX_SEQ_LENGTH="${BAR_V5_MAX_SEQ_LENGTH:-8192}" \
  PER_DEVICE_TRAIN_BATCH_SIZE="${BAR_V5_PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  GRADIENT_ACCUMULATION_STEPS="${BAR_V5_GRADIENT_ACCUMULATION_STEPS:-16}" \
  LEARNING_RATE="${BAR_V5_LEARNING_RATE:-5e-7}" \
  NUM_TRAIN_EPOCHS="${BAR_V5_NUM_TRAIN_EPOCHS:-1}" \
  SAVE_STEPS="${BAR_V5_SAVE_STEPS:-100}" \
  SAVE_TOTAL_LIMIT="${BAR_V5_SAVE_TOTAL_LIMIT:-2}" \
  bash scripts/run_bar_exam_v5_sft_train_guarded.sh 2>&1 | tee "$LOG_DIR/bar_exam_v5_sft.train.log"

if [ ! -d "$BAR_V5_OUTPUT_DIR/final_full" ]; then
  echo "bar v5 final checkpoint missing: $BAR_V5_OUTPUT_DIR/final_full" >&2
  exit 1
fi

upload_background \
  "$BAR_V5_OUTPUT_DIR/final_full" \
  "$BAR_V5_HUB_MODEL_ID" \
  "Upload KO-CPT repair BarExamV5 SFT final_full" \
  "bar_exam_v5_sft"

if [ "$WAIT_FOR_UPLOADS" = "1" ]; then
  wait_uploads
fi

echo "chain_finished_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
