#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

RUN_ID="${RUN_ID:-20260630_stage2_gate_limit50}"
LIMIT="${LIMIT:-50}"
GPU_COUNT="${GPU_COUNT:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
OUT_ROOT="${OUT_ROOT:-/home/work/.data/lfm2_ko_sft/eval}"

echo "stage2_gate_eval_start=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "run_id=$RUN_ID"
echo "limit=$LIMIT"

MODELS_FILE=configs/eval_models_stage2_gate_20260630.txt \
TASK_GROUPS_FILE=configs/eval_task_groups_stage2_gate_20260630.txt \
RUN_ID="$RUN_ID" \
LIMIT="$LIMIT" \
GPU_COUNT="$GPU_COUNT" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
OUT_ROOT="$OUT_ROOT" \
bash scripts/run_vllm_eval_8gpu_queue.sh

echo "stage2_gate_eval_done=$(TZ=Asia/Seoul date '+%F %T KST')"
