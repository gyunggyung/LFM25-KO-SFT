#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
CPT_WORK_DIR="$ROOT_DIR/lfm2_ko_cpt"
HRM_DIR="$ROOT_DIR/HRM-Text"
CONFIG="${CONFIG:-$WORK_DIR/configs/sft_sources_20260628.json}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_sft_prep}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/home/work/.data/hrm_text_prepared/kohrm_sft_lfm25_terminal_toolbench_full_v1/tokenizer.json}"
EPOCHS="${EPOCHS:-3}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8193}"

mkdir -p "$WORK_DIR/logs" "$DATA_ROOT/manifests" "$DATA_ROOT/prepared/jsonl_converted" "$DATA_ROOT/prepared/tokenized_jsonl" "$DATA_ROOT/prepared/merged"
pids=()

echo "run_id=$RUN_ID"
echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "config=$CONFIG"

inventory_log="$WORK_DIR/logs/${RUN_ID}_inventory.log"
python "$WORK_DIR/scripts/inventory_sft_sources.py" \
  --config "$CONFIG" \
  --output "$DATA_ROOT/manifests/${RUN_ID}_inventory.json" \
  > "$inventory_log" 2>&1 &
pid=$!
pids+=("$pid")
echo "inventory_pid=$pid log=$inventory_log"

convert_one() {
  local id="$1"
  local input="$2"
  local out="$DATA_ROOT/prepared/jsonl_converted/${id}.sft.jsonl"
  local log="$WORK_DIR/logs/${RUN_ID}_convert_${id}.log"
  python "$WORK_DIR/scripts/convert_lfm_text_jsonl_to_sft_jsonl.py" \
    --input "$input" \
    --output "$out" \
    --source-id "$id" \
    --condition direct \
    --dedupe \
    > "$log" 2>&1
  echo "$out"
}

for spec in \
  "legal_source_agent:/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/006_ko_legal_source_agent_sft_20260621.jsonl" \
  "legal_rag_round15:/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/007_ko_legal_rag_agent_sft_round15_v2.jsonl" \
  "current_law_bar_json:/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/008_current_law_bar_json_answer_sft_20260621.jsonl"
do
  id="${spec%%:*}"
  input="${spec#*:}"
  log="$WORK_DIR/logs/${RUN_ID}_jsonl_prepare_${id}.log"
  (
    out="$(convert_one "$id" "$input")"
    python "$HRM_DIR/scripts/prepare_sft_data.py" \
      --train "$out" \
      --tokenizer "$TOKENIZER_JSON" \
      --output "$DATA_ROOT/prepared/tokenized_jsonl/${id}" \
      --epochs "$EPOCHS" \
      --context-size "$CONTEXT_SIZE" \
      --overflow-policy truncate-instruction-middle \
      --truncate-head-tokens 1024 \
      --strip-think-blocks \
      --condition-override direct \
      --progress-interval 1000
) > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "jsonl_prepare_$id pid=$pid log=$log"
done

merge_8k_log="$WORK_DIR/logs/${RUN_ID}_merge_8k_terminal.log"
(
  python "$HRM_DIR/scripts/merge_prepared_sft_data.py" \
    --inputs \
      /home/work/.data/hrm_text_prepared/kohrm_sft_lfm25_terminal_toolbench_full_v1 \
    --output "$DATA_ROOT/prepared/merged/${RUN_ID}_stage1_8k_terminal_toolbench" \
    --epochs "$EPOCHS" \
    --seed 60628 \
    --copy-tokenizer
) > "$merge_8k_log" 2>&1 &
pid=$!
pids+=("$pid")
echo "merge_8k_terminal_pid=$pid log=$merge_8k_log"

merge_4k_log="$WORK_DIR/logs/${RUN_ID}_merge_4k_core.log"
(
  python "$HRM_DIR/scripts/merge_prepared_sft_data.py" \
    --inputs \
      /home/work/.data/hrm_text_prepared/kohrm_sft_behavior_core_v1 \
      /home/work/.data/hrm_text_prepared/sft_swe_zero_v1 \
      /home/work/.data/hrm_text_prepared/korean_legal_tasks_full_v1 \
      /home/work/.data/hrm_text_prepared/sft_bcai_finance_kor_v1 \
      /home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_clean_duckdb_v1 \
    --output "$DATA_ROOT/prepared/merged/${RUN_ID}_stage2_4k_ko_legal_finance_coding_core" \
    --epochs "$EPOCHS" \
    --seed 60628 \
    --copy-tokenizer
) > "$merge_4k_log" 2>&1 &
pid=$!
pids+=("$pid")
echo "merge_4k_core_pid=$pid log=$merge_4k_log"

download_log="$WORK_DIR/logs/${RUN_ID}_downloads.log"
bash "$WORK_DIR/scripts/run_external_downloads.sh" > "$download_log" 2>&1 &
pid=$!
pids+=("$pid")
echo "downloads_pid=$pid log=$download_log"

echo "started_sft_preprocess_parallel"

failed=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    echo "job_done pid=$pid"
  else
    rc=$?
    echo "job_failed pid=$pid rc=$rc" >&2
    failed=1
  fi
done

echo "finished_sft_preprocess_parallel time_kst=$(TZ=Asia/Seoul date '+%F %T KST') failed=$failed"
exit "$failed"
