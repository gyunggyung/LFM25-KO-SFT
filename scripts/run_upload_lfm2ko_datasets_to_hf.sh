#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

LOG_DIR=logs/hf_upload
mkdir -p "$LOG_DIR"

echo "dataset_upload_start=$(TZ=Asia/Seoul date '+%F %T KST')"
python scripts/create_agentic_model_repo_card.py
python scripts/upload_lfm2ko_datasets_to_hf.py "$@"
echo "dataset_upload_done=$(TZ=Asia/Seoul date '+%F %T KST')"
