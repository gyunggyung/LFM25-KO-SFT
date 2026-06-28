#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
HRM_DIR="$ROOT_DIR/HRM-Text"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PATH="${MODEL_PATH:-/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0b-finance-text2sql-20260628/final_full}"
BASE_STAGE2="${BASE_STAGE2:-$DATA_ROOT/prepared/lfm_chat/20260628_lfmchat_stage2_diverse_ko_swe_reasoning_4k}"
RUN_ID="${RUN_ID:-20260628_lfmchat_stage2_plus_kotsqa}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"

JSONL_ROOT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}.jsonl"
PART_ROOT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}.parts"
KOTSQA_PART="$PART_ROOT/kotsqa_v2_train"
OUT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}_4k"

mkdir -p "$WORK_DIR/logs/prep" "$JSONL_ROOT" "$PART_ROOT" "$OUT"

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "run_id=$RUN_ID"
echo "base_stage2=$BASE_STAGE2"
echo "output=$OUT"

if [ ! -d "$BASE_STAGE2" ]; then
  echo "missing_base_stage2=$BASE_STAGE2" >&2
  exit 1
fi

"$VENV/bin/python" "$WORK_DIR/scripts/convert_kotsqa_to_lfm_sft_jsonl.py" \
  --dataset etri-lirs/KoTSQA-v.2.0 \
  --split train \
  --output "$JSONL_ROOT/kotsqa_v2_train.jsonl" \
  --source-id etri_lirs_kotsqa_v2_train \
  --max-passages 4 \
  --max-passage-chars 1800 \
  > "$WORK_DIR/logs/prep/${RUN_ID}_kotsqa_convert.log" 2>&1

"$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
  --train "$JSONL_ROOT/kotsqa_v2_train.jsonl" \
  --tokenizer "$MODEL_PATH/tokenizer.json" \
  --output "$KOTSQA_PART" \
  --epochs 1 \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --dedupe \
  --source-id etri_lirs_kotsqa_v2_train \
  --progress-interval 2000 \
  > "$WORK_DIR/logs/prep/${RUN_ID}_kotsqa_lfm_prepare.log" 2>&1

"$VENV/bin/python" "$HRM_DIR/scripts/merge_prepared_sft_data.py" \
  --inputs "$BASE_STAGE2" "$KOTSQA_PART" \
  --output "$OUT" \
  --epochs 1 \
  --seed 60628 \
  --copy-tokenizer

echo "done_stage2_plus_kotsqa=$OUT"
