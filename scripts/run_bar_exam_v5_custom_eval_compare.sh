#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
OUT_DIR="${OUT_DIR:-/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_custom_$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${LIMIT:-12}"
mkdir -p "$OUT_DIR/logs"

declare -a PIDS=()

launch_eval() {
  local gpu="$1"
  local port="$2"
  local label="$3"
  local model="$4"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    PORT="$port" \
    LABEL="$label" \
    SERVED_MODEL_NAME="$label" \
    MODEL_ID="$model" \
    OUT_DIR="$OUT_DIR" \
    LIMIT="$LIMIT" \
    bash "$WORK_DIR/scripts/run_bar_exam_v5_custom_eval_server.sh"
  ) >"$OUT_DIR/logs/${label}.wrapper.log" 2>&1 &
  PIDS+=("$!")
  echo "launched label=$label gpu=$gpu port=$port model=$model"
}

launch_eval 0 11601 ko_cpt /home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full
launch_eval 1 11602 repair_sft /home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630/final_full
launch_eval 2 11603 bar_v5_sft /home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT-20260630/final_full

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done

"$WORK_DIR/scripts/summarize_bar_exam_v5_custom_eval.py" "$OUT_DIR" > "$OUT_DIR/SUMMARY.md"
echo "summary=$OUT_DIR/SUMMARY.md"
echo "failed_jobs=$failed"
if [ "$failed" -gt 0 ]; then
  exit 1
fi
