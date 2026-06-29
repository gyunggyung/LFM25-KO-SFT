#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

LOG_DIR=logs/train
mkdir -p "$LOG_DIR" logs/eval
LOG="$LOG_DIR/20260630_agentic_fable_chain_after_stage2.log"

STAGE2_FINAL="${STAGE2_FINAL:-/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full}"
AGENTIC_DATA="${AGENTIC_DATA:-/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260630_lfmchat_agentic_fable_grounded_8k}"
AGENTIC_OUT="${AGENTIC_OUT:-/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-Agentic-SFT-fable-grounded-20260630}"
AGENTIC_HUB_MODEL_ID="${AGENTIC_HUB_MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT}"

echo "agentic_chain_wait_stage2_start=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
echo "stage2_final=$STAGE2_FINAL" | tee -a "$LOG"

while [ ! -d "$STAGE2_FINAL" ]; do
  sleep 90
done

echo "stage2_final_seen=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"

if [ "${RUN_STAGE2_GATE_EVAL:-1}" = "1" ]; then
  echo "stage2_gate_eval_launch=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
  RUN_ID="${STAGE2_GATE_RUN_ID:-20260630_stage2_gate_limit50}" \
  LIMIT="${STAGE2_GATE_LIMIT:-50}" \
  GPU_COUNT=8 \
  MAX_MODEL_LEN=8192 \
  bash scripts/run_stage2_gate_eval_quick.sh 2>&1 | tee -a "$LOG"
fi

if [ ! -d "$AGENTIC_DATA" ]; then
  echo "agentic_prepare_launch=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
  TOKENIZER_MODEL_PATH="$STAGE2_FINAL" \
  RUN_ID=20260630_lfmchat_agentic_fable_grounded \
  MAX_SEQ_LENGTH=8192 \
  bash scripts/run_prepare_lfmchat_agentic_fable_grounded.sh 2>&1 | tee -a "$LOG"
fi

echo "agentic_train_launch=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
DATASET_PATH="$AGENTIC_DATA" \
MODEL_PATH="$STAGE2_FINAL" \
RUN_ID=agentic-fable-grounded-20260630 \
STAGE_NAME=stage3_agentic_fable_grounded \
MAX_SEQ_LENGTH=8192 \
OUTPUT_DIR="$AGENTIC_OUT" \
HUB_MODEL_ID="$AGENTIC_HUB_MODEL_ID" \
PER_DEVICE_TRAIN_BATCH_SIZE="${AGENTIC_PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
GRADIENT_ACCUMULATION_STEPS="${AGENTIC_GRADIENT_ACCUMULATION_STEPS:-16}" \
LEARNING_RATE="${AGENTIC_LEARNING_RATE:-1e-6}" \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS="${AGENTIC_MAX_STEPS:--1}" \
SAVE_STEPS="${AGENTIC_SAVE_STEPS:-500}" \
SAVE_TOTAL_LIMIT=2 \
LOGGING_STEPS=10 \
DATALOADER_NUM_WORKERS=0 \
PUSH_TO_HUB=1 \
MASTER_PORT="${AGENTIC_MASTER_PORT:-29515}" \
bash scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh > "$LOG_DIR/20260630_agentic_fable_grounded.launch.log" 2>&1

echo "agentic_train_done=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"

if [ "${RUN_AGENTIC_FINAL_EVAL:-1}" = "1" ]; then
  echo "agentic_final_lm_eval_launch=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
  RUN_ID="${AGENTIC_FINAL_RUN_ID:-20260630_agentic_final_limit50}" \
  LIMIT="${AGENTIC_FINAL_LIMIT:-50}" \
  MODELS_FILE=configs/eval_models_agentic_final_20260630.txt \
  TASK_GROUPS_FILE=configs/eval_task_groups_agentic_final_20260630.txt \
  GPU_COUNT=8 \
  MAX_MODEL_LEN=8192 \
  OUT_ROOT=/home/work/.data/lfm2_ko_sft/eval \
  bash scripts/run_vllm_eval_8gpu_queue.sh 2>&1 | tee -a "$LOG"

  echo "agentic_smoke_eval_launch=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
  MODEL_ID="$AGENTIC_OUT/final_full" \
  SERVED_MODEL_NAME=lfm2-ko-agentic-sft \
  CUDA_VISIBLE_DEVICES=0 \
  PORT=1053 \
  bash scripts/run_agentic_fable_eval_smoke.sh 2>&1 | tee -a "$LOG"
fi

echo "agentic_chain_done=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
