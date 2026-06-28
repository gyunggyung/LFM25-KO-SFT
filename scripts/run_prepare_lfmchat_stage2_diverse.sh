#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
HRM_DIR="$ROOT_DIR/HRM-Text"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PATH="${MODEL_PATH:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
RUN_ID="${RUN_ID:-20260628_lfmchat_stage2_diverse_ko_swe_reasoning}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
CONCURRENCY="${CONCURRENCY:-4}"

JSONL_ROOT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}.jsonl"
PREP_ROOT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}.parts"
OUT="$DATA_ROOT/prepared/lfm_chat/${RUN_ID}_4k"

mkdir -p "$WORK_DIR/logs/prep" "$JSONL_ROOT" "$PREP_ROOT" "$OUT"

# Source format:
# id|prepared_path|max_sample_len|target_tokens
# target_tokens=0 means full dataset. Raw CPT-style corpora are intentionally
# excluded from this stage.
SOURCES=(
  "korean_domain_core|/home/work/.data/hrm_text_prepared/kohrm_sft_korean_domain_core_v1|4096|0"
  "behavior_core|/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_core_v1|4096|0"
  "swe_zero|/home/work/.data/hrm_text_prepared/sft_swe_zero_v1|4096|0"
  "swe_glm_mix|/home/work/.data/hrm_text_prepared/sft_swe_glm_mix_v1|4096|0"
  "swe_zero_30m|/home/work/.data/hrm_text_prepared/kohrm_sft_comp_swe_zero_30m_v1|4096|0"
  "glm_reasoning|/home/work/.data/hrm_text_prepared/sft_glm_reasoning_v1|4096|0"
  "hf_extra_reasoning_agent_mm|/home/work/.data/hrm_text_prepared/hf_extra_reasoning_agent_mm_v1|4096|0"
  "agent_reasoning_25m|/home/work/.data/hrm_text_prepared/kohrm_sft_comp_agent_reasoning_25m_v1|4096|0"
  "finance_50m|/home/work/.data/hrm_text_prepared/kohrm_sft_comp_finance_50m_v1|4096|0"
  "korean_legal_50m|/home/work/.data/hrm_text_prepared/kohrm_sft_comp_korean_legal_50m_v1|4096|0"
  "text2sql_duckdb|/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_clean_duckdb_v1|4096|0"
)

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "run_id=$RUN_ID"
echo "max_seq_length=$MAX_SEQ_LENGTH"
echo "concurrency=$CONCURRENCY"
echo "output=$OUT"

run_one() {
  local spec="$1"
  IFS='|' read -r id input max_sample_len target_tokens <<< "$spec"
  local jsonl="$JSONL_ROOT/${id}.jsonl"
  local part="$PREP_ROOT/$id"
  local export_log="$WORK_DIR/logs/prep/${RUN_ID}_${id}_export.log"
  local prep_log="$WORK_DIR/logs/prep/${RUN_ID}_${id}_lfm_prepare.log"

  if [ ! -d "$input" ]; then
    echo "missing_input id=$id path=$input" >&2
    return 1
  fi

  "$VENV/bin/python" "$WORK_DIR/scripts/export_prepared_to_lfm_jsonl.py" \
    --input "$input" \
    --output "$jsonl" \
    --source-id "$id" \
    --max-sample-len "$max_sample_len" \
    --target-tokens "$target_tokens" \
    --seed 60628 \
    --progress-interval 10000 \
    > "$export_log" 2>&1

  "$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
    --train "$jsonl" \
    --tokenizer "$MODEL_PATH/tokenizer.json" \
    --output "$part" \
    --epochs 1 \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --strip-think-blocks \
    --dedupe \
    --source-id "$id" \
    --progress-interval 10000 \
    > "$prep_log" 2>&1

  echo "prepared id=$id part=$part"
}

pids=()
failed=0
for spec in "${SOURCES[@]}"; do
  run_one "$spec" &
  pids+=("$!")
  if [ "${#pids[@]}" -ge "$CONCURRENCY" ]; then
    if ! wait -n; then
      failed=1
    fi
    alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive+=("$pid")
      fi
    done
    pids=("${alive[@]}")
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "stage2_prep_failed" >&2
  exit 1
fi

inputs=()
for spec in "${SOURCES[@]}"; do
  IFS='|' read -r id _ <<< "$spec"
  inputs+=("$PREP_ROOT/$id")
done

python "$HRM_DIR/scripts/merge_prepared_sft_data.py" \
  --inputs "${inputs[@]}" \
  --output "$OUT" \
  --epochs 1 \
  --seed 60628 \
  --copy-tokenizer

echo "done_stage2_diverse=$OUT"
