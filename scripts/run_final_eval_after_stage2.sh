#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

STAGE2_FINAL=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full
LOG=logs/eval/20260630_final_sft_lm_eval_chain.log
mkdir -p logs/eval

echo "final_eval_wait_start=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"

while [ ! -d "$STAGE2_FINAL" ]; do
  sleep 180
done

echo "final_eval_start=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"

RUN_ID=20260630_final_sft_lm_eval \
MODELS_FILE=configs/eval_models_final_sft_20260628.txt \
TASK_GROUPS_FILE=configs/eval_task_groups_final_sft_lm_eval_20260628.txt \
GPU_COUNT=8 \
MAX_MODEL_LEN=8192 \
OUT_ROOT=/home/work/.data/lfm2_ko_sft/eval \
bash scripts/run_vllm_eval_8gpu_queue.sh | tee -a "$LOG"

echo "final_eval_done=$(TZ=Asia/Seoul date '+%F %T KST')" | tee -a "$LOG"
