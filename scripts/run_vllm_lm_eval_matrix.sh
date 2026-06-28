#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"
OUT_ROOT="${OUT_ROOT:-/home/work/.data/lfm2_ko_sft/eval}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_vllm_lm_eval}"

MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
MODEL_LABEL="${MODEL_LABEL:-$(basename "$MODEL_ID" | tr '/:' '__')}"
TASKS="${TASKS:-ifeval,gsm8k,arc_challenge,hellaswag,truthfulqa_mc2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
LIMIT="${LIMIT:-}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-0}"

if [ ! -x "$VLLM_ENV/bin/python" ]; then
  echo "Missing VLLM_ENV python: $VLLM_ENV/bin/python" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

if [ -f "$ROOT_DIR/.env" ] && [ -z "${HF_TOKEN:-}" ]; then
  hf_token_line="$("$WORK_DIR/scripts/print_hf_token_from_env.py" "$ROOT_DIR/.env" 2>/dev/null || true)"
  if [ -n "$hf_token_line" ]; then
    export HF_TOKEN="$hf_token_line"
  fi
fi

OUT_DIR="$OUT_ROOT/$RUN_ID/$MODEL_LABEL"
mkdir -p "$OUT_DIR"

MODEL_ARGS="pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=$TENSOR_PARALLEL_SIZE,gpu_memory_utilization=$GPU_MEMORY_UTILIZATION,max_model_len=$MAX_MODEL_LEN"

EXTRA_ARGS=()
if [ -n "$LIMIT" ]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [ "$FEWSHOT_AS_MULTITURN" = "1" ]; then
  EXTRA_ARGS+=(--fewshot_as_multiturn)
fi

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "model_id=$MODEL_ID"
echo "model_label=$MODEL_LABEL"
echo "tasks=$TASKS"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "max_model_len=$MAX_MODEL_LEN"
echo "out_dir=$OUT_DIR"

"$VLLM_ENV/bin/python" -m lm_eval \
  --model vllm \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUT_DIR" \
  --log_samples \
  "${EXTRA_ARGS[@]}"

"$WORK_DIR/scripts/summarize_lm_eval_results.py" "$OUT_ROOT/$RUN_ID" > "$OUT_ROOT/$RUN_ID/SUMMARY.md"
echo "summary=$OUT_ROOT/$RUN_ID/SUMMARY.md"
