# LFM2.5 KO SFT 별도 평가 하네스

작성 시각: 2026-06-28

## 목적

`lm-eval`에 없는 Liquid 공식 카드 계열 benchmark를 최종 SFT 학습 완료 뒤 바로
돌릴 수 있게 준비한다. 학습이 끝날 때까지 GPU는 학습 체인이 우선이고, 이 문서의
외부 하네스는 CPU/network 설치를 미리 끝낸 뒤 최종 모델이 나오면 GPU를 평가로
넘긴다.

## 공식 기준과 로컬 상태

| benchmark | upstream | local status | runner |
|---|---|---|---|
| BFCLv3/BFCLv4 | <https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard> | setup/runner prepared | `scripts/run_bfcl_v4_openai_endpoint.sh` |
| Tau2 Telecom/Retail | <https://github.com/sierra-research/tau2-bench> | setup/runner prepared | `scripts/run_tau2_openai_endpoint.sh` |
| IFBench | <https://github.com/allenai/IFBench> | setup/generate/eval runner prepared | `scripts/run_ifbench_openai_endpoint.sh` |
| Multi-IF | <https://github.com/facebookresearch/Multi-IF> | setup/runner/data clone prepared | `scripts/run_multi_if_vllm.sh` |
| AA-Omniscience | <https://huggingface.co/datasets/ArtificialAnalysis/AA-Omniscience-Public> | LightEval env prepared; task string must be confirmed after install | `scripts/run_aa_omniscience_lighteval.sh` |

Liquid base model/card reference:

- <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>
- <https://www.liquid.ai/blog/lfm2-5-8b-a1b>

## 설치

설치는 학습과 병렬로 CPU/network만 사용한다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/setup_external_eval_harnesses.sh
```

설치 위치:

```text
/home/work/.data/lfm2_ko_sft/eval_harnesses
```

생성되는 주요 경로:

```text
/home/work/.data/lfm2_ko_sft/eval_harnesses/repos/gorilla/berkeley-function-call-leaderboard
/home/work/.data/lfm2_ko_sft/eval_harnesses/repos/tau2-bench
/home/work/.data/lfm2_ko_sft/eval_harnesses/repos/IFBench
/home/work/.data/lfm2_ko_sft/eval_harnesses/repos/Multi-IF
/home/work/.data/lfm2_ko_sft/eval_harnesses/repos/Multi-IF/data/Multi-IF
/home/work/.data/lfm2_ko_sft/eval_harnesses/repos/lighteval
```

## 공통 vLLM OpenAI 서버

외부 하네스는 OpenAI-compatible endpoint를 우선 사용한다. 최종 학습이 끝나고 GPU가
비면 다음처럼 서버를 띄운다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
CUDA_VISIBLE_DEVICES=0 \
MODEL_ID=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full \
SERVED_MODEL_NAME=lfm2-ko-sft \
PORT=1053 \
MAX_MODEL_LEN=8192 \
bash scripts/start_vllm_openai_server_for_harness.sh
```

여러 하네스를 병렬로 돌릴 때는 GPU별로 `PORT`와 `CUDA_VISIBLE_DEVICES`를 다르게
잡는다.

## BFCLv4

BFCL은 `bfcl generate` 후 `bfcl evaluate` 순서다. 공식 README 기준으로
pre-existing OpenAI-compatible endpoint는 `--skip-server-setup`을 쓴다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
LOCAL_SERVER_ENDPOINT=localhost \
LOCAL_SERVER_PORT=1053 \
REMOTE_OPENAI_BASE_URL=http://localhost:1053/v1 \
REMOTE_OPENAI_API_KEY=EMPTY \
BFCL_MODEL_NAME=lfm2-ko-sft-FC \
bash scripts/run_bfcl_v4_openai_endpoint.sh
```

주의: BFCL의 공식 점수로 제출하려면 LFM tool-call 출력 parser/handler가 BFCL 모델
mapping에 정확히 등록되어야 한다. 현재 runner는 endpoint 경로와 결과 저장 구조를
준비한 것이고, 최종 공식 표에는 handler 확인 후 산출된 점수만 올린다.

## Tau2 Telecom/Retail

Tau2는 `uv sync` 환경에서 `tau2 run`을 호출한다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
OPENAI_BASE_URL=http://localhost:1053/v1 \
OPENAI_API_KEY=EMPTY \
MODEL_NAME=lfm2-ko-sft \
DOMAINS="telecom retail" \
NUM_TRIALS=1 \
NUM_TASKS=20 \
bash scripts/run_tau2_openai_endpoint.sh
```

빠른 smoke는 `NUM_TASKS=5`, 최종 표는 domain별 full/base split을 사용한다.

## IFBench

IFBench는 공식 `generate_responses.py`로 completion JSONL을 만들고 `run_eval.py`로
채점한다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
API_BASE=http://localhost:1053/v1 \
API_KEY=EMPTY \
MODEL_NAME=lfm2-ko-sft \
WORKERS=8 \
MAX_TOKENS=2048 \
bash scripts/run_ifbench_openai_endpoint.sh
```

결과는 다음 아래에 쌓인다.

```text
/home/work/.data/lfm2_ko_sft/eval/external_harnesses/ifbench
```

## Multi-IF

Multi-IF는 공식 vLLM runner를 사용한다. 데이터는 setup 스크립트가
`facebook/Multi-IF` HF dataset을 clone한다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full \
TOKENIZER_PATH=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full \
BATCH_SIZE=8 \
TENSOR_PARALLEL_SIZE=1 \
bash scripts/run_multi_if_vllm.sh
```

## AA-Omniscience

LightEval 환경만 먼저 준비한다. 설치 후 task registry에서 실제 task string을 확인한
뒤 고정해야 한다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_aa_omniscience_lighteval.sh
```

최종 실행 전 확인 항목:

```bash
/home/work/.data/lfm2_ko_sft/eval_harnesses/venvs/aa-omniscience/bin/lighteval --help
/home/work/.data/lfm2_ko_sft/eval_harnesses/venvs/aa-omniscience/bin/lighteval tasks list | grep -i omniscience
```

## 자동 순서

1. 현재 full SFT 체인이 GPU 8장을 계속 사용한다.
2. `scripts/run_final_eval_after_stage2.sh`가 Stage2 완료를 감시한다.
3. Stage2 final model이 생기면 `lm-eval`/vLLM full queue를 8 GPU에 바로 분배한다.
4. 외부 하네스는 `lm-eval` 결과를 모델 카드에 반영한 뒤, 필요한 GPU 수만큼 vLLM
   endpoint를 띄워 BFCL/Tau2/IFBench/Multi-IF 순서로 돌린다.

## 결과 반영 위치

- README: `Quick Eval Snapshot`, `Evaluation Plan`, `External Harnesses`
- 모델 카드: 상단 성능표와 하네스별 raw result 경로
- 최종 결과: `/home/work/.data/lfm2_ko_sft/eval/.../SUMMARY.md`
- 외부 결과: `/home/work/.data/lfm2_ko_sft/eval/external_harnesses/...`
