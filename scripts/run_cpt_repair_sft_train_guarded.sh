#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_TRAIN:-0}" != "1" ]; then
  cat <<'MSG'
Refusing to start training.

This launcher is intentionally guarded because the current instruction is:
do not use GPUs and do not train.

To run later after explicit approval:
  ALLOW_TRAIN=1 bash scripts/run_cpt_repair_sft_train_guarded.sh
MSG
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"

export MODEL_PATH="${MODEL_PATH:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
export DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/prepared/repair_cpt/20260630_cpt_mcqa_repair_4k/lfm_chat_4k}"
export RUN_ID="${RUN_ID:-cpt-repair-mcqa-answer-format-20260630}"
export STAGE_NAME="${STAGE_NAME:-cpt_repair_mcqa_answer_format}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630}"
export HUB_MODEL_ID="${HUB_MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export PUSH_TO_HUB="${PUSH_TO_HUB:-}"

if [ ! -d "$DATASET_PATH" ]; then
  echo "Prepared dataset missing: $DATASET_PATH" >&2
  echo "Run scripts/run_prepare_cpt_repair_sft_full.sh first." >&2
  exit 1
fi

cd "$WORK_DIR"
bash scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh

