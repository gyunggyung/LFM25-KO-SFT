#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

BACKEND="${AGENT_BACKEND:-vllm}"
if [ "$BACKEND" = "llamacpp" ]; then
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8080/v1}"
  export MODEL_NAME="${MODEL_NAME:-lfm2-ko-sft-gguf}"
else
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:1053/v1}"
  export MODEL_NAME="${MODEL_NAME:-lfm2-ko-sft}"
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

exec python agent_harness/lfm2ko_agent.py \
  --backend "$BACKEND" \
  --workspace /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft \
  "$@"
