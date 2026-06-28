#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-/home/work/.data/lfm2_ko_sft/eval_harnesses}"
AA_VENV="$HARNESS_ROOT/venvs/aa-omniscience"
MODEL_ID="${MODEL_ID:-LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT}"
OUT_DIR="${OUT_DIR:-/home/work/.data/lfm2_ko_sft/eval/external_harnesses/aa_omniscience}"

mkdir -p "$OUT_DIR"

if [ ! -x "$AA_VENV/bin/lighteval" ]; then
  echo "lighteval venv missing. Run scripts/setup_external_eval_harnesses.sh first." >&2
  exit 2
fi

cat >&2 <<EOF
AA-Omniscience is available in Hugging Face lighteval source as a dedicated task
backed by ArtificialAnalysis/AA-Omniscience-Public, but the local lighteval CLI
task string can vary by version. This script verifies the environment and leaves
the model/task command explicit for the final run.

Model: $MODEL_ID
Output: $OUT_DIR

Next final-run command should be filled after checking:
  $AA_VENV/bin/lighteval --help
  $AA_VENV/bin/lighteval tasks list | grep -i omniscience
EOF

"$AA_VENV/bin/lighteval" --help >/dev/null
