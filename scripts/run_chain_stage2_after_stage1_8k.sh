#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
LOG_DIR=logs/train
mkdir -p "$LOG_DIR"

STAGE1_8K_FINAL=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage1-8k-legal-terminal-20260628/final_full
STAGE2_DATA=/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage2_plus_kotsqa_4k
STAGE2_FALLBACK=/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage2_diverse_ko_swe_reasoning_4k

echo "chain_stage2_wait_start=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG_DIR/20260628_chain_stage2_after_stage1_8k.log"

while [ ! -d "$STAGE1_8K_FINAL" ]; do
  sleep 120
done

if [ ! -d "$STAGE2_DATA" ]; then
  echo "stage2_plus_kotsqa_missing_using_fallback=$STAGE2_FALLBACK" | tee -a "$LOG_DIR/20260628_chain_stage2_after_stage1_8k.log"
  STAGE2_DATA="$STAGE2_FALLBACK"
fi

DATASET_PATH="$STAGE2_DATA" \
MODEL_PATH="$STAGE1_8K_FINAL" \
RUN_ID=stage2-4k-diverse-kotsqa-20260628 \
STAGE_NAME=stage2_4k_diverse_kotsqa \
MAX_SEQ_LENGTH=4096 \
OUTPUT_DIR=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
LEARNING_RATE=2e-6 \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS=-1 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=2 \
LOGGING_STEPS=10 \
DATALOADER_NUM_WORKERS=0 \
PUSH_TO_HUB=1 \
MASTER_PORT=29514 \
bash scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh > "$LOG_DIR/20260628_stage2_4k_diverse_kotsqa.launch.log" 2>&1
