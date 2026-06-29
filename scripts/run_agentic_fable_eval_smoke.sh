#!/usr/bin/env bash
set -euo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

MODEL_ID="${MODEL_ID:-/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-Agentic-SFT-fable-grounded-20260630/final_full}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-lfm2-ko-agentic-sft}"
PORT="${PORT:-1053}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-logs/agent_eval}"
OUT_DIR="${OUT_DIR:-/home/work/.data/lfm2_ko_sft/eval/20260630_agentic_smoke}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "agentic_smoke_start=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "model_id=$MODEL_ID"
echo "port=$PORT"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MODEL_ID="$MODEL_ID" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
PORT="$PORT" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}" \
bash scripts/start_vllm_openai_server_for_harness.sh > "$LOG_DIR/20260630_agentic_vllm_server.log" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

python - "$PORT" <<'PY'
import json
import sys
import time
import urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 900
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(resp.read().decode("utf-8")[:500])
            raise SystemExit(0)
    except Exception as exc:
        last = exc
        time.sleep(10)
raise SystemExit(f"vllm server did not become ready: {last}")
PY

MODE=real \
AGENT_BACKEND=vllm \
OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" \
OPENAI_API_KEY=EMPTY \
MODEL_NAME="$SERVED_MODEL_NAME" \
bash scripts/run_lfm2ko_agent_smoke_eval.sh > "$OUT_DIR/agent_harness_smoke.log" 2>&1

OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" \
OPENAI_API_KEY=EMPTY \
MODEL_NAME="$SERVED_MODEL_NAME" \
bash scripts/run_lfm2ko_agent_harness.sh \
  --context-window 8192 \
  --prompt-budget 22000 \
  "학습 로그를 보면 step/planned/loss/examples_per_sec가 있다. README.md와 docs/RUNBOOK_20260628.ko.md를 참고해서 현재 모델 학습-평가 체인을 점검하는 절차를 한국어로 설명해라." \
  > "$OUT_DIR/agent_harness_grounded_prompt.log" 2>&1

MODE=real \
AGENT_BACKEND=vllm \
OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" \
OPENAI_API_KEY=EMPTY \
MODEL_NAME="$SERVED_MODEL_NAME" \
OUT_DIR="$OUT_DIR" \
RUN_ID=agentic_eval_suite \
EXECUTE_TOOLS=1 \
ALLOW_SHELL=1 \
bash scripts/run_lfm2ko_agentic_eval.sh > "$OUT_DIR/agentic_eval_suite.log" 2>&1

echo "agentic_smoke_done=$(TZ=Asia/Seoul date '+%F %T KST')"
echo "out_dir=$OUT_DIR"
