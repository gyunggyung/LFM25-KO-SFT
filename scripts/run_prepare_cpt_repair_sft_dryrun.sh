#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/prepared/repair_cpt/20260630_cpt_mcqa_repair_dryrun}"
RAW_DIR="$OUT_ROOT/raw"
PREP_DIR="$OUT_ROOT/lfm_chat_4k"
REPORT_DIR="$OUT_ROOT/reports"

LEGAL_JSONL="${LEGAL_JSONL:-/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/008_current_law_bar_json_answer_sft_20260621.jsonl}"
KOTSQA_JSONL="${KOTSQA_JSONL:-/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage2_plus_kotsqa.jsonl/kotsqa_v2_train.jsonl}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full/tokenizer.json}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_LEGAL_ROWS="${MAX_LEGAL_ROWS:-200}"
MAX_KOTSQA_ROWS="${MAX_KOTSQA_ROWS:-100}"

mkdir -p "$RAW_DIR" "$PREP_DIR" "$REPORT_DIR"

RAW_JSONL="$RAW_DIR/cpt_repair_mcqa_kotsqa_dryrun.jsonl"

echo "dryrun_start=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "raw_jsonl=$RAW_JSONL"
echo "prep_dir=$PREP_DIR"
echo "tokenizer_json=$TOKENIZER_JSON"

"$VENV/bin/python" "$WORK_DIR/scripts/build_cpt_repair_sft_jsonl.py" \
  --legal-bar-jsonl "$LEGAL_JSONL" \
  --kotsqa-jsonl "$KOTSQA_JSONL" \
  --output "$RAW_JSONL" \
  --max-legal-rows "$MAX_LEGAL_ROWS" \
  --max-kotsqa-rows "$MAX_KOTSQA_ROWS"

"$VENV/bin/python" "$WORK_DIR/scripts/validate_cpt_repair_sft_jsonl.py" \
  --input "$RAW_JSONL" \
  --report "$REPORT_DIR/raw_validation.json" \
  --fail-on-warning

"$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
  --train "$RAW_JSONL" \
  --tokenizer "$TOKENIZER_JSON" \
  --output "$PREP_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --dedupe \
  --source-id "cpt_repair_dryrun_20260630" \
  --progress-interval 1000

"$VENV/bin/python" "$WORK_DIR/scripts/validate_prepared_sft_arrays.py" \
  --dataset-path "$PREP_DIR" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --report "$REPORT_DIR/prepared_validation.json" \
  --expect-max-seq-length "$MAX_SEQ_LENGTH"

echo "dryrun_done=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "reports=$REPORT_DIR"

