#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"

MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
LABEL="${LABEL:?LABEL is required}"
PORT="${PORT:?PORT is required}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
OUT_DIR="${OUT_DIR:-/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_custom_$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${LIMIT:-12}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-9000}"
MAX_TOKENS="${MAX_TOKENS:-256}"
STRICT_ONE_TOKEN="${STRICT_ONE_TOKEN:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"

mkdir -p "$OUT_DIR/logs"
server_log="$OUT_DIR/logs/${LABEL}.server.log"
eval_log="$OUT_DIR/logs/${LABEL}.eval.log"

export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

"$VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "vLLM server did not become ready for $LABEL; log=$server_log" >&2
  exit 1
fi

strict_args=()
if [ "$STRICT_ONE_TOKEN" = "1" ]; then
  strict_args+=(--strict-one-token)
fi

"$VLLM_ENV/bin/python" "$WORK_DIR/scripts/run_bar_exam_v5_custom_eval.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "$SERVED_MODEL_NAME" \
  --label "$LABEL" \
  --out-dir "$OUT_DIR" \
  --limit "$LIMIT" \
  --max-context-chars "$MAX_CONTEXT_CHARS" \
  --max-tokens "$MAX_TOKENS" \
  "${strict_args[@]}" \
  >"$eval_log" 2>&1

cat "$OUT_DIR/${LABEL}.summary.json"
