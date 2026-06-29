#!/usr/bin/env bash
set -euo pipefail

MODEL_GGUF="${MODEL_GGUF:?MODEL_GGUF is required, e.g. /path/to/LFM2.5-8B-A1B-KO-SFT-Q8_0.gguf}"
LLAMACPP_SERVER="${LLAMACPP_SERVER:-llama-server}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-8192}"
THREADS="${THREADS:-$(nproc)}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-128}"
MODEL_ALIAS="${MODEL_ALIAS:-lfm2-ko-sft-gguf}"

exec "$LLAMACPP_SERVER" \
  --model "$MODEL_GGUF" \
  --alias "$MODEL_ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH_SIZE"
