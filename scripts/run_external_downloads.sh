#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"
VENV="${VENV:-$ROOT_DIR/.liquid-sft-env}"
mkdir -p "$DATA_ROOT/downloads" "$DATA_ROOT/downloads/hf_models"

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"

repo_dir="$DATA_ROOT/downloads/LLM-Ko-Datasets"
if [ -d "$repo_dir/.git" ]; then
  echo "updating $repo_dir"
  git -C "$repo_dir" pull --ff-only || true
else
  echo "cloning LLM-Ko-Datasets"
  git clone --depth 1 https://github.com/gyunggyung/LLM-Ko-Datasets "$repo_dir" || true
fi

if [ -f "$ROOT_DIR/.env" ] && [ -z "${HF_TOKEN:-}" ]; then
  hf_token_line="$("$VENV/bin/python" "$WORK_DIR/scripts/print_hf_token_from_env.py" "$ROOT_DIR/.env" 2>/dev/null || true)"
  if [ -n "$hf_token_line" ]; then
    export HF_TOKEN="$hf_token_line"
  fi
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  echo "hf snapshot metadata/model card"
  HF_HOME="${HF_HOME:-/home/work/.data/huggingface}" \
  huggingface-cli download LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
    README.md config.json tokenizer.json tokenizer_config.json \
    --local-dir "$DATA_ROOT/downloads/hf_models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch" \
    --local-dir-use-symlinks False || true
else
  echo "huggingface-cli not found; skipping hf metadata download"
fi

echo "downloads_done_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
