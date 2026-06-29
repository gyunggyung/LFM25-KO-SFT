#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

MODE="${MODE:-mock}"
BACKEND="${AGENT_BACKEND:-vllm}"
OUT_DIR="${OUT_DIR:-/home/work/.data/lfm2_ko_sft/eval/agentic_harness}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_agentic_harness}"
mkdir -p "$OUT_DIR"

ARGS=(
  --tasks "${TASKS:-agent_harness/agentic_eval_tasks.jsonl}"
  --output "$OUT_DIR/${RUN_ID}.jsonl"
  --workspace /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
  --context-window "${AGENT_CONTEXT_WINDOW:-8192}"
  --prompt-budget "${AGENT_PROMPT_BUDGET_CHARS:-24000}"
  --max-turns "${AGENT_MAX_TURNS:-3}"
  --max-tokens "${AGENT_MAX_TOKENS:-2048}"
)

if [ "$MODE" = "mock" ]; then
  ARGS+=(--mock --allow-failures)
else
  ARGS+=(--backend "$BACKEND")
  if [ "$BACKEND" = "llamacpp" ]; then
    ARGS+=(--endpoint "${OPENAI_BASE_URL:-http://localhost:8080/v1}")
    ARGS+=(--model "${MODEL_NAME:-lfm2-ko-agentic-sft-gguf}")
  else
    ARGS+=(--endpoint "${OPENAI_BASE_URL:-http://localhost:1053/v1}")
    ARGS+=(--model "${MODEL_NAME:-lfm2-ko-agentic-sft}")
  fi
  ARGS+=(--api-key "${OPENAI_API_KEY:-EMPTY}")
fi

if [ "${ALLOW_SHELL:-0}" = "1" ]; then
  ARGS+=(--allow-shell)
fi
if [ "${EXECUTE_TOOLS:-0}" = "1" ]; then
  ARGS+=(--execute-tools)
fi
if [ "${ALLOW_WRITE:-0}" = "1" ]; then
  ARGS+=(--allow-write)
fi
if [ "${ALLOW_FAILURES:-0}" = "1" ]; then
  ARGS+=(--allow-failures)
fi
if [ "${PRINT_ANSWERS:-0}" = "1" ]; then
  ARGS+=(--print-answers)
fi

python agent_harness/agentic_eval.py "${ARGS[@]}"
