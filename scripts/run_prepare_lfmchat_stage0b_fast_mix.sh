#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
HRM_DIR="$ROOT_DIR/HRM-Text"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PATH="${MODEL_PATH:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
RUN_ID="${RUN_ID:-20260628_lfmchat_stage0b_fast_mix}"
PREP_ROOT="$DATA_ROOT/prepared/lfm_chat/$RUN_ID.parts"
OUT="$DATA_ROOT/prepared/lfm_chat/$RUN_ID"

mkdir -p "$WORK_DIR/logs/prep" "$PREP_ROOT" "$OUT"

prepare_one() {
  local id="$1"
  local input="$2"
  local max_rows="$3"
  local max_seq="$4"
  local log="$WORK_DIR/logs/prep/${RUN_ID}_${id}.log"
  "$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
    --train "$input" \
    --tokenizer "$MODEL_PATH/tokenizer.json" \
    --output "$PREP_ROOT/$id" \
    --epochs 1 \
    --max-seq-length "$max_seq" \
    --max-rows "$max_rows" \
    --strip-think-blocks \
    --dedupe \
    --source-id "$id" \
    --progress-interval 10000 \
    > "$log" 2>&1
}

pids=()
prepare_one finance_bcai_120k /home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/001_bcai_finance_kor_hrm_20260524.jsonl 120000 4096 &
pids+=("$!")
prepare_one text2sql_160k /home/work/.data/hrm_text_raw/text2sql/text2sql_core_clean_sft.jsonl 160000 4096 &
pids+=("$!")
prepare_one legal_tasks_40000 /home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/004_korean_legal_tasks_full_20260524.jsonl 40000 8192 &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    echo "prep_done pid=$pid"
  else
    rc=$?
    echo "prep_failed pid=$pid rc=$rc" >&2
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  exit "$failed"
fi

python "$HRM_DIR/scripts/merge_prepared_sft_data.py" \
  --inputs \
    "$PREP_ROOT/finance_bcai_120k" \
    "$PREP_ROOT/text2sql_160k" \
    "$PREP_ROOT/legal_tasks_40000" \
  --output "$OUT" \
  --epochs 1 \
  --seed 60628 \
  --copy-tokenizer

echo "$OUT"
