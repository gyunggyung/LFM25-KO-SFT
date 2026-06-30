#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/prepared/repair_cpt/20260630_cpt_mcqa_repair_4k}"
RAW_DIR="$OUT_ROOT/raw"
EXPORT_DIR="$RAW_DIR/preservation_exports"
PREP_DIR="$OUT_ROOT/lfm_chat_4k"
REPORT_DIR="$OUT_ROOT/reports"

LEGAL_JSONL="${LEGAL_JSONL:-/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/008_current_law_bar_json_answer_sft_20260621.jsonl}"
KOTSQA_JSONL="${KOTSQA_JSONL:-/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage2_plus_kotsqa.jsonl/kotsqa_v2_train.jsonl}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full/tokenizer.json}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
TARGET_TOKENS="${TARGET_TOKENS:-200000000}"

mkdir -p "$RAW_DIR" "$EXPORT_DIR" "$PREP_DIR" "$REPORT_DIR"

REPAIR_JSONL="$RAW_DIR/cpt_repair_mcqa_kotsqa_full.jsonl"

echo "full_preprocess_start=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "target_tokens=$TARGET_TOKENS"
echo "out_root=$OUT_ROOT"

"$VENV/bin/python" "$WORK_DIR/scripts/build_cpt_repair_sft_jsonl.py" \
  --legal-bar-jsonl "$LEGAL_JSONL" \
  --kotsqa-jsonl "$KOTSQA_JSONL" \
  --output "$REPAIR_JSONL"

"$VENV/bin/python" "$WORK_DIR/scripts/validate_cpt_repair_sft_jsonl.py" \
  --input "$REPAIR_JSONL" \
  --report "$REPORT_DIR/raw_repair_validation.json" \
  --fail-on-warning

export_source() {
  local source_id="$1"
  local input_dir="$2"
  local target_tokens="$3"
  local output="$EXPORT_DIR/${source_id}.jsonl"
  echo "export_source=$source_id target_tokens=$target_tokens"
  "$VENV/bin/python" "$WORK_DIR/scripts/export_prepared_to_lfm_jsonl.py" \
    --input "$input_dir" \
    --output "$output" \
    --source-id "$source_id" \
    --target-tokens "$target_tokens" \
    --max-sample-len "$MAX_SEQ_LENGTH"
}

export_source finance_compact /home/work/.data/hrm_text_prepared/kohrm_sft_comp_finance_50m_v1 30000000
export_source text2sql_duckdb /home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_clean_duckdb_v1 20000000
export_source swe_compact /home/work/.data/hrm_text_prepared/kohrm_sft_comp_swe_zero_30m_v1 25000000
export_source reasoning_compact /home/work/.data/hrm_text_prepared/kohrm_sft_comp_glm_reasoning_20m_v1 20000000
export_source agent_reasoning_compact /home/work/.data/hrm_text_prepared/kohrm_sft_comp_agent_reasoning_25m_v1 15000000
export_source behavior_mini /home/work/.data/hrm_text_prepared/kohrm_sft_behavior_mini_v1 10000000

"$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
  --train "$REPAIR_JSONL" "$EXPORT_DIR" \
  --tokenizer "$TOKENIZER_JSON" \
  --output "$PREP_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --dedupe \
  --target-tokens "$TARGET_TOKENS" \
  --source-id "cpt_repair_full_20260630" \
  --progress-interval 10000

"$VENV/bin/python" "$WORK_DIR/scripts/validate_prepared_sft_arrays.py" \
  --dataset-path "$PREP_DIR" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --report "$REPORT_DIR/prepared_validation.json" \
  --expect-max-seq-length "$MAX_SEQ_LENGTH"

echo "full_preprocess_done=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "prepared_dataset=$PREP_DIR"
echo "reports=$REPORT_DIR"

