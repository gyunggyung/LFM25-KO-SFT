#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/lfm2_ko_sft"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/lfm2_ko_sft}"

echo "time_kst=$(TZ=Asia/Seoul date '+%F %T KST')"
echo
echo "[sessions]"
tmux list-sessions -F '#S' 2>/dev/null | rg '^lfm2ko_sft|^lfm2ko_eval' || true
echo
echo "[gpu]"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo
echo "[prepared outputs]"
find "$DATA_ROOT/prepared" -maxdepth 3 -type f \( -name 'metadata.json' -o -name '*stats.json' -o -name 'merge_stats.json' \) -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -40
echo
echo "[logs tail]"
for f in "$WORK_DIR"/logs/*; do
  [ -f "$f" ] || continue
  echo "== $(basename "$f") =="
  tail -3 "$f" || true
done
