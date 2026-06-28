#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_lfmchat_sft}"

MODEL_PATH="${MODEL_PATH:-/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0-legal-20260628/final_full}"
DATASET_PATH="${DATASET_PATH:?DATASET_PATH is required}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-SFT-$RUN_ID}"
HUB_MODEL_ID="${HUB_MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT}"
STAGE_NAME="${STAGE_NAME:-lfmchat_sft}"

mkdir -p "$WORK_DIR/logs/train" "$OUTPUT_DIR"

if [ -f "$ROOT_DIR/.env" ] && [ -z "${HF_TOKEN:-}" ]; then
  hf_token_line="$("$VENV/bin/python" "$WORK_DIR/scripts/print_hf_token_from_env.py" "$ROOT_DIR/.env" 2>/dev/null || true)"
  if [ -n "$hf_token_line" ]; then
    export HF_TOKEN="$hf_token_line"
  fi
fi

export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "run_id=$RUN_ID"
echo "stage_name=$STAGE_NAME"
echo "model_path=$MODEL_PATH"
echo "dataset_path=$DATASET_PATH"
echo "max_seq_length=$MAX_SEQ_LENGTH"
echo "output_dir=$OUTPUT_DIR"
echo "hub_model_id=$HUB_MODEL_ID"
echo "dataloader_num_workers=${DATALOADER_NUM_WORKERS:-0}"
if [ -n "${DEEPSPEED_CONFIG:-}" ]; then
  echo "deepspeed_config=$DEEPSPEED_CONFIG"
fi

extra_train_args=()
if [ -n "${DEEPSPEED_CONFIG:-}" ]; then
  extra_train_args+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

"$VENV/bin/accelerate" launch --num_processes 8 --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  "$WORK_DIR/scripts/train_lfm25_ko_sft_prepared.py" \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --stage-name "$STAGE_NAME" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning-rate "${LEARNING_RATE:-6e-6}" \
  --num-train-epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --max-steps "${MAX_STEPS:--1}" \
  --save-steps "${SAVE_STEPS:-1000}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging-steps "${LOGGING_STEPS:-5}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-0}" \
  --push-to-hub \
  --hub-model-id "$HUB_MODEL_ID" \
  "${extra_train_args[@]}"
