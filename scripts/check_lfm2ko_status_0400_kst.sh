#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

OUT_DIR=logs/watch
OUT="$OUT_DIR/20260629_0400_kst_status.md"
mkdir -p "$OUT_DIR"

{
  echo "# LFM2 KO SFT 04:00 KST Status"
  echo
  echo "- checked_at: $(TZ=Asia/Seoul date '+%F %T KST')"
  echo
  echo "## tmux"
  echo '```text'
  tmux ls 2>/dev/null | grep -E 'lfm2ko|sft' || true
  echo '```'
  echo
  echo "## GPU"
  echo '```text'
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
  echo '```'
  echo
  echo "## Stage1 4k Recent Train Log"
  echo '```json'
  tail -n 20 /home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage1-4k-finance-text2sql-20260628/train_log.jsonl 2>/dev/null || true
  echo '```'
  echo
  echo "## Checkpoints"
  echo '```text'
  find /home/work/.data/lfm2_ko_sft/models -maxdepth 2 -type d \
    \( -name 'checkpoint-*' -o -name 'final_full' \) \
    -path '*LFM2.5-8B-A1B-KO-SFT-stage*' \
    -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -40 || true
  echo '```'
  echo
  echo "## Storage"
  echo '```text'
  df -h /home/work /home/work/.data || true
  echo '```'
} > "$OUT"

echo "wrote_status=$OUT"
