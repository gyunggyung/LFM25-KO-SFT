#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
MODE="${MODE:-context_solver}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
TARGET_TOKENS="${TARGET_TOKENS:-0}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full/tokenizer.json}"

OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/prepared/bar_exam_v5/20260630_bar_exam_v5_${MODE}_${MAX_SEQ_LENGTH}}"
RAW_DIR="$OUT_ROOT/raw"
REPORT_DIR="$OUT_ROOT/reports"
PREP_DIR="$OUT_ROOT/lfm_chat_${MAX_SEQ_LENGTH}"
RAW_JSONL="$RAW_DIR/bar_exam_v5_${MODE}.jsonl"
STATS_JSON="$REPORT_DIR/build_stats.json"

CURRENT_LAW_SIMPLE="$WORK_DIR/data/current_law_bar_exam_sft_1000/hf_dataset/sft/train.jsonl"
CURRENT_LAW_HARD="$WORK_DIR/data/current_law_bar_exam_hard_sft_1000/hf_dataset/sft/train.jsonl"

mkdir -p "$RAW_DIR" "$REPORT_DIR" "$PREP_DIR"

echo "prepare_start=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "mode=$MODE"
echo "max_seq_length=$MAX_SEQ_LENGTH"
echo "raw_jsonl=$RAW_JSONL"
echo "prepared_dir=$PREP_DIR"

"$VENV/bin/python" "$WORK_DIR/scripts/build_bar_exam_v5_sft_jsonl.py" \
  --bar-exam-root "$WORK_DIR/data/bar_exam" \
  --current-law-jsonl "$CURRENT_LAW_SIMPLE" \
  --current-law-jsonl "$CURRENT_LAW_HARD" \
  --mode "$MODE" \
  --output "$RAW_JSONL" \
  --stats "$STATS_JSON"

"$VENV/bin/python" "$WORK_DIR/scripts/validate_bar_exam_v5_sft_jsonl.py" \
  --input "$RAW_JSONL" \
  --report "$REPORT_DIR/raw_validation.json" \
  --mode "$MODE"

PREP_ARGS=(
  --train "$RAW_JSONL"
  --tokenizer "$TOKENIZER_JSON"
  --output "$PREP_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --dedupe
  --source-id "bar_exam_v5_${MODE}_20260630"
  --progress-interval 1000
)

if [ "$TARGET_TOKENS" != "0" ]; then
  PREP_ARGS+=(--target-tokens "$TARGET_TOKENS")
fi

"$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" "${PREP_ARGS[@]}"

"$VENV/bin/python" "$WORK_DIR/scripts/validate_prepared_sft_arrays.py" \
  --dataset-path "$PREP_DIR" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --report "$REPORT_DIR/prepared_validation.json" \
  --expect-max-seq-length "$MAX_SEQ_LENGTH"

echo "prepare_done=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "raw_validation=$REPORT_DIR/raw_validation.json"
echo "prepared_validation=$REPORT_DIR/prepared_validation.json"
