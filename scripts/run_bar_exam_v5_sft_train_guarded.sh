#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_TRAIN:-0}" != "1" ]; then
  cat <<'MSG'
Refusing to start training.

Current instruction is to prepare CPU-side data/code only and not use GPUs.

To train later after explicit approval:
  ALLOW_TRAIN=1 bash scripts/run_bar_exam_v5_sft_train_guarded.sh
MSG
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"

REPAIR_MODEL_DEFAULT="$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630/final_full"
CPT_MODEL_DEFAULT="/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full"

if [ -z "${MODEL_PATH:-}" ]; then
  if [ -d "$REPAIR_MODEL_DEFAULT" ]; then
    export MODEL_PATH="$REPAIR_MODEL_DEFAULT"
  elif [ "${ALLOW_BASELINE_CPT_START:-0}" = "1" ]; then
    export MODEL_PATH="$CPT_MODEL_DEFAULT"
  else
    cat <<MSG >&2
Repair checkpoint is missing:
  $REPAIR_MODEL_DEFAULT

This v5 context SFT is intended to run after the CPT repair SFT.
Set MODEL_PATH explicitly, or set ALLOW_BASELINE_CPT_START=1 to start from KO-CPT.
MSG
    exit 1
  fi
fi

export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/prepared/bar_exam_v5/20260630_bar_exam_v5_context_solver_8192/lfm_chat_8192}"
export RUN_ID="${RUN_ID:-bar-exam-v5-context-grounded-20260630}"
export STAGE_NAME="${STAGE_NAME:-bar_exam_v5_context_grounded}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT-20260630}"
export HUB_MODEL_ID="${HUB_MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-7}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-250}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-}"

if [ ! -d "$DATASET_PATH" ]; then
  echo "Prepared dataset missing: $DATASET_PATH" >&2
  echo "Run scripts/run_prepare_bar_exam_v5_sft.sh first." >&2
  exit 1
fi

cd "$WORK_DIR"
bash scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh
