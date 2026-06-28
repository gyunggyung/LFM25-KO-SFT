#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
CHAIN_ID="${CHAIN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_ko_sft_chain}"
STAGE1_OUTPUT="$DATA_ROOT/models/LFM2.5-8B-A1B-KO-SFT-stage1-$CHAIN_ID"
STAGE2_OUTPUT="$DATA_ROOT/models/LFM2.5-8B-A1B-KO-SFT-stage2-$CHAIN_ID"

mkdir -p "$WORK_DIR/logs/train"

RUN_ID="stage1-$CHAIN_ID" OUTPUT_DIR="$STAGE1_OUTPUT" \
  bash "$WORK_DIR/scripts/run_lfm25_ko_sft_stage1.sh" \
  2>&1 | tee "$WORK_DIR/logs/train/${CHAIN_ID}_stage1.log"

RUN_ID="stage2-$CHAIN_ID" MODEL_PATH="$STAGE1_OUTPUT/final_full" OUTPUT_DIR="$STAGE2_OUTPUT" \
  bash "$WORK_DIR/scripts/run_lfm25_ko_sft_stage2.sh" \
  2>&1 | tee "$WORK_DIR/logs/train/${CHAIN_ID}_stage2.log"

echo "$STAGE2_OUTPUT/final_full" > "$DATA_ROOT/models/latest_final_path.txt"
echo "chain_done final=$STAGE2_OUTPUT/final_full"
