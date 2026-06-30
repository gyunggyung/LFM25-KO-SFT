#!/usr/bin/env bash
set -euo pipefail

if [ "${ALLOW_EVAL:-0}" != "1" ]; then
  cat <<'MSG'
Refusing to start evaluation.

This launcher starts vLLM/lm-eval jobs on GPUs. Run only after explicit approval:
  ALLOW_EVAL=1 bash scripts/run_repair_bar_v5_eval_after_training.sh
MSG
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/eval}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"

REPAIR_MODEL="${REPAIR_MODEL:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630/final_full}"
BAR_V5_MODEL="${BAR_V5_MODEL:-$DATA_ROOT/models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT-20260630/final_full}"

if [ ! -d "$REPAIR_MODEL" ]; then
  echo "Missing repair model: $REPAIR_MODEL" >&2
  exit 1
fi
if [ ! -d "$BAR_V5_MODEL" ]; then
  echo "Missing bar v5 model: $BAR_V5_MODEL" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "repair_model=$REPAIR_MODEL"
echo "bar_v5_model=$BAR_V5_MODEL"
echo "out_root=$OUT_ROOT"

RUN_ID="repair_sft_gate_$RUN_STAMP" \
OUT_ROOT="$OUT_ROOT" \
MODELS_FILE="$WORK_DIR/configs/eval_models_repair_sft_20260630.txt" \
TASK_GROUPS_FILE="$WORK_DIR/configs/eval_task_groups_repair_sft_gate_20260630.txt" \
GPU_IDS="${REPAIR_GPU_IDS:-0,1,2,3}" \
FAIL_ON_EVAL_ERROR="${FAIL_ON_EVAL_ERROR:-0}" \
bash "$WORK_DIR/scripts/run_vllm_eval_8gpu_queue.sh" >"$OUT_ROOT/repair_sft_gate_$RUN_STAMP.log" 2>&1 &
repair_pid="$!"

RUN_ID="bar_exam_v5_sft_gate_$RUN_STAMP" \
OUT_ROOT="$OUT_ROOT" \
MODELS_FILE="$WORK_DIR/configs/eval_models_bar_exam_v5_sft_20260630.txt" \
TASK_GROUPS_FILE="$WORK_DIR/configs/eval_task_groups_bar_exam_v5_gate_20260630.txt" \
GPU_IDS="${BAR_V5_GPU_IDS:-4,5,6,7}" \
FAIL_ON_EVAL_ERROR="${FAIL_ON_EVAL_ERROR:-0}" \
bash "$WORK_DIR/scripts/run_vllm_eval_8gpu_queue.sh" >"$OUT_ROOT/bar_exam_v5_sft_gate_$RUN_STAMP.log" 2>&1 &
bar_pid="$!"

failed=0
if wait "$repair_pid"; then
  echo "repair_eval_finished status=0"
else
  failed=$((failed + 1))
  echo "repair_eval_finished status=failed" >&2
fi

if wait "$bar_pid"; then
  echo "bar_v5_eval_finished status=0"
else
  failed=$((failed + 1))
  echo "bar_v5_eval_finished status=failed" >&2
fi

echo "repair_summary=$OUT_ROOT/repair_sft_gate_$RUN_STAMP/SUMMARY.md"
echo "bar_v5_summary=$OUT_ROOT/bar_exam_v5_sft_gate_$RUN_STAMP/SUMMARY.md"
echo "failed_eval_groups=$failed"

if [ "${FAIL_ON_EVAL_ERROR:-0}" = "1" ] && [ "$failed" -gt 0 ]; then
  exit 1
fi
