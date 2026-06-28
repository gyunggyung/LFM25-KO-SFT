#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-/home/work/.data/lfm2_ko_sft/eval_harnesses}"
MULTI_IF_ROOT="$HARNESS_ROOT/repos/Multi-IF"
MULTI_IF_VENV="$HARNESS_ROOT/venvs/multi-if"
MODEL_PATH="${MODEL_PATH:-LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
INPUT_DATA_CSV="${INPUT_DATA_CSV:-$MULTI_IF_ROOT/data/Multi-IF/Multi-IF.csv}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

if [ ! -x "$MULTI_IF_VENV/bin/python" ]; then
  echo "Multi-IF venv missing. Run scripts/setup_external_eval_harnesses.sh first." >&2
  exit 2
fi
if [ ! -f "$INPUT_DATA_CSV" ]; then
  echo "Multi-IF data missing at $INPUT_DATA_CSV. Clone/download facebook/Multi-IF dataset first." >&2
  exit 2
fi

cd "$MULTI_IF_ROOT"
"$MULTI_IF_VENV/bin/python" multi_turn_instruct_following_eval_vllm.py \
  --model_path "$MODEL_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --input_data_csv "$INPUT_DATA_CSV" \
  --batch_size "$BATCH_SIZE" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
