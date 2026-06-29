#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
HRM_DIR="$ROOT_DIR/HRM-Text"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
FABLE_DIR="${FABLE_DIR:-$ROOT_DIR/fable_distillation/datasets_ko}"
RUN_ID="${RUN_ID:-20260630_lfmchat_agentic_fable_grounded}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"

TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-}"
if [ -z "$TOKENIZER_MODEL_PATH" ]; then
  for candidate in \
    "$DATA_ROOT/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full" \
    "$DATA_ROOT/models/LFM2.5-8B-A1B-KO-SFT-stage1-8k-legal-terminal-20260628/final_full" \
    "$DATA_ROOT/models/LFM2.5-8B-A1B-KO-SFT-stage1-4k-finance-text2sql-20260628/final_full" \
    "$DATA_ROOT/models/LFM2.5-8B-A1B-KO-SFT-stage0b-finance-text2sql-20260628/final_full" \
    "/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full"; do
    if [ -f "$candidate/tokenizer.json" ]; then
      TOKENIZER_MODEL_PATH="$candidate"
      break
    fi
  done
fi

if [ -z "$TOKENIZER_MODEL_PATH" ] || [ ! -f "$TOKENIZER_MODEL_PATH/tokenizer.json" ]; then
  echo "missing_tokenizer_model_path=$TOKENIZER_MODEL_PATH" >&2
  exit 2
fi

JSONL_ROOT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}.jsonl"
PART_ROOT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}.parts"
FABLE_JSONL="$JSONL_ROOT/fable5_helio_agentic.jsonl"
GROUNDING_JSONL="$JSONL_ROOT/local_grounding_logs_docs.jsonl"
FABLE_PART="$PART_ROOT/fable5_helio_agentic"
GROUNDING_PART="$PART_ROOT/local_grounding_logs_docs"
OUT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}_8k"

mkdir -p "$WORK_DIR/logs/prep" "$JSONL_ROOT" "$PART_ROOT" "$OUT"

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "run_id=$RUN_ID"
echo "tokenizer_model_path=$TOKENIZER_MODEL_PATH"
echo "fable_dir=$FABLE_DIR"
echo "max_seq_length=$MAX_SEQ_LENGTH"
echo "output=$OUT"

"$VENV/bin/python" "$WORK_DIR/scripts/convert_fable_agentic_to_lfm_jsonl.py" \
  --input "$FABLE_DIR/fable5_ko_sft_20260624.jsonl" "$FABLE_DIR/helio_ko_sft_20260628.jsonl" \
  --output "$FABLE_JSONL" \
  --strip-think \
  --dedupe \
  --keep-overclaim \
  > "$WORK_DIR/logs/prep/${RUN_ID}_fable_convert.log" 2>&1

"$VENV/bin/python" "$WORK_DIR/scripts/build_agentic_grounding_sft_jsonl.py" \
  --workspace "$WORK_DIR" \
  --data-root "$DATA_ROOT" \
  --output "$GROUNDING_JSONL" \
  > "$WORK_DIR/logs/prep/${RUN_ID}_grounding_build.log" 2>&1

"$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
  --train "$FABLE_JSONL" \
  --tokenizer "$TOKENIZER_MODEL_PATH/tokenizer.json" \
  --output "$FABLE_PART" \
  --epochs 1 \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --dedupe \
  --source-id fable5_helio_agentic \
  --progress-interval 1000 \
  > "$WORK_DIR/logs/prep/${RUN_ID}_fable_lfm_prepare.log" 2>&1

"$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
  --train "$GROUNDING_JSONL" \
  --tokenizer "$TOKENIZER_MODEL_PATH/tokenizer.json" \
  --output "$GROUNDING_PART" \
  --epochs 1 \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --dedupe \
  --source-id local_agentic_grounding \
  --progress-interval 1000 \
  > "$WORK_DIR/logs/prep/${RUN_ID}_grounding_lfm_prepare.log" 2>&1

"$VENV/bin/python" "$HRM_DIR/scripts/merge_prepared_sft_data.py" \
  --inputs "$FABLE_PART" "$GROUNDING_PART" \
  --output "$OUT" \
  --epochs 1 \
  --seed 60630 \
  --copy-tokenizer

echo "done_agentic_fable_grounded=$OUT"
cat "$OUT/merge_stats.json" 2>/dev/null || true
