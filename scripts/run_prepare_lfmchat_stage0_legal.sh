#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PATH="${MODEL_PATH:-/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full}"
RUN_ID="${RUN_ID:-20260628_lfmchat_stage0_legal}"
OUT="$DATA_ROOT/prepared/lfm_chat/$RUN_ID"

mkdir -p "$WORK_DIR/logs/prep" "$OUT"

"$VENV/bin/python" "$WORK_DIR/scripts/prepare_lfm_chat_sft_data.py" \
  --train \
    "$DATA_ROOT/prepared/jsonl_converted/legal_source_agent.sft.jsonl" \
    "$DATA_ROOT/prepared/jsonl_converted/legal_rag_round15.sft.jsonl" \
    "$DATA_ROOT/prepared/jsonl_converted/current_law_bar_json.sft.jsonl" \
  --tokenizer "$MODEL_PATH/tokenizer.json" \
  --output "$OUT" \
  --epochs 1 \
  --max-seq-length 8192 \
  --strip-think-blocks \
  --dedupe \
  --source-id legal_source_rag_bar_lfm_tokenizer \
  --progress-interval 1000

echo "$OUT"
