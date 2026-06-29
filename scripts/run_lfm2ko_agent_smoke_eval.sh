#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

MODE="${MODE:-mock}"
ARGS=(
  --workspace /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
  --smoke
)

if [ "$MODE" = "mock" ]; then
  ARGS+=(--mock)
else
  BACKEND="${AGENT_BACKEND:-vllm}"
  ARGS+=(--backend "$BACKEND")
  if [ "$BACKEND" = "llamacpp" ]; then
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8080/v1}"
    export MODEL_NAME="${MODEL_NAME:-lfm2-ko-sft-gguf}"
  else
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:1053/v1}"
    export MODEL_NAME="${MODEL_NAME:-lfm2-ko-sft}"
  fi
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
fi

python agent_harness/lfm2ko_agent.py "${ARGS[@]}"
