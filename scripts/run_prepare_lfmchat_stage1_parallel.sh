#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
HRM_DIR="$ROOT_DIR/HRM-Text"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PATH="${MODEL_PATH:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
RUN_ID="${RUN_ID:-20260628_lfmchat_stage1_ko_finance_terminal_text2sql}"
PREP_ROOT="$DATA_ROOT/prepared/lfm_chat/$RUN_ID.parts"
OUT_4K="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}_4k_finance_text2sql"
OUT_8K="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}_8k_legal_terminal"

mkdir -p "$WORK_DIR/logs/prep" "$PREP_ROOT" "$OUT_4K" "$OUT_8K"

prepare_one() {
  local id="$1"
  local input="$2"
  local max_seq="$3"
  local log="$WORK_DIR/logs/prep/${RUN_ID}_${id}.log"
  "$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
    --train "$input" \
    --tokenizer "$MODEL_PATH/tokenizer.json" \
    --output "$PREP_ROOT/$id" \
    --epochs 1 \
    --max-seq-length "$max_seq" \
    --strip-think-blocks \
    --dedupe \
    --source-id "$id" \
    --progress-interval 20000 \
    > "$log" 2>&1
}

pids=()
prepare_one korean_legal_tasks /home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/004_korean_legal_tasks_full_20260524.jsonl 8192 &
pids+=("$!")
prepare_one finance_bcai_hrm /home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/001_bcai_finance_kor_hrm_20260524.jsonl 4096 &
pids+=("$!")
prepare_one terminal_toolbench /home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/009_lfm25_terminal_toolbench_hrm_turns_v1.jsonl 8192 &
pids+=("$!")
prepare_one text2sql_core /home/work/.data/hrm_text_raw/text2sql/text2sql_core_clean_sft.jsonl 4096 &
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
    "$PREP_ROOT/finance_bcai_hrm" \
    "$PREP_ROOT/text2sql_core" \
  --output "$OUT_4K" \
  --epochs 1 \
  --seed 60628 \
  --copy-tokenizer

python "$HRM_DIR/scripts/merge_prepared_sft_data.py" \
  --inputs \
    "$PREP_ROOT/korean_legal_tasks" \
    "$PREP_ROOT/terminal_toolbench" \
  --output "$OUT_8K" \
  --epochs 1 \
  --seed 60628 \
  --copy-tokenizer

echo "$OUT_4K"
echo "$OUT_8K"
