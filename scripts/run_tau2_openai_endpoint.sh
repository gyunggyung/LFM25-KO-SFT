#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-/home/work/.data/lfm2_ko_sft/eval_harnesses}"
TAU2_ROOT="$HARNESS_ROOT/repos/tau2-bench"
PORT="${PORT:-1053}"
MODEL_NAME="${MODEL_NAME:-lfm2-ko-sft}"
AGENT_LLM="${AGENT_LLM:-openai/${MODEL_NAME}}"
USER_LLM="${USER_LLM:-openai/${MODEL_NAME}}"
DOMAINS="${DOMAINS:-telecom retail}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_TASKS="${NUM_TASKS:-20}"

if [ ! -d "$TAU2_ROOT/.venv" ]; then
  echo "tau2-bench uv environment missing. Run scripts/setup_external_eval_harnesses.sh first." >&2
  exit 2
fi

cd "$TAU2_ROOT"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:${PORT}/v1}"

for domain in $DOMAINS; do
  uv run tau2 run \
    --domain "$domain" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --num-trials "$NUM_TRIALS" \
    --num-tasks "$NUM_TASKS"
done
