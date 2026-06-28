#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
LOG_DIR=logs/train
mkdir -p "$LOG_DIR"

STAGE1_4K_SESSION="${STAGE1_4K_SESSION:-lfm2ko_sft_stage1_4k_20260628}"
STAGE1_4K_FINAL=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage1-4k-finance-text2sql-20260628/final_full

echo "chain_stage1_8k_wait_start=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG_DIR/20260628_chain_stage1_8k_after_stage1_4k.log"

while tmux has-session -t "$STAGE1_4K_SESSION" 2>/dev/null; do
  sleep 120
done

if [ ! -d "$STAGE1_4K_FINAL" ]; then
  echo "stage1_4k final_full missing; not chaining" | tee -a "$LOG_DIR/20260628_chain_stage1_8k_after_stage1_4k.log"
  exit 1
fi

DATASET_PATH=/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage1_ko_finance_terminal_text2sql_8k_legal_terminal \
MODEL_PATH="$STAGE1_4K_FINAL" \
RUN_ID=stage1-8k-legal-terminal-20260628 \
STAGE_NAME=stage1_8k_legal_terminal \
MAX_SEQ_LENGTH=8192 \
OUTPUT_DIR=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage1-8k-legal-terminal-20260628 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
LEARNING_RATE=3e-6 \
NUM_TRAIN_EPOCHS=1 \
MAX_STEPS=-1 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=2 \
LOGGING_STEPS=10 \
DATALOADER_NUM_WORKERS=0 \
PUSH_TO_HUB=1 \
MASTER_PORT=29513 \
bash scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh > "$LOG_DIR/20260628_stage1_8k_legal_terminal.launch.log" 2>&1
