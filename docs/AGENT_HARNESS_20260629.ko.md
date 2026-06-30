# LFM2.5 KO SFT 전용 Agent Harness

작성 시각: 2026-06-29

## 목적

최종 `LFM2.5-8B-A1B-KO-SFT` 모델을 Codex/Claude Code처럼 범용 자율 에이전트로
무작정 쓰는 대신, 학습 데이터에서 강하게 반복된 행동만 안정적으로 쓰기 위한
bounded harness다.

2026-06-30 기준 추가 GPU 학습/평가는 중지했다. 이 harness 문서는 코드 구조와
CPU/mock 검증 방법을 보존하기 위한 참고 문서다.

## 잘 하도록 설계한 일

| profile | 잘하는 일 | 이유 |
|---|---|---|
| `ko_legal` | 한국어 법률 QA, 변시형 정답 형식, 근거 기반 설명 | legal/bar/RAG/source SFT 데이터 |
| `ko_finance` | 한국어 금융/회계 지표 설명, 표 기반 계산 | finance SFT 데이터 |
| `terminal_tool` | 터미널 상태 요약, 로그/파일 읽기, tool-call 형식 | Terminal/ToolBench/LFM tool-use 데이터 |
| `text2sql` | DuckDB/Text2SQL 쿼리 작성 | Text2SQL clean DuckDB 데이터 |
| `code_assistant` | 작은 코드 수정 계획, 테스트 명령, 파일 경로 중심 설명 | SWE/coding/terminal mix |
| `general_ko_instruction` | 일반 한국어 지시 따르기와 요약 | Korean instruction mix |

## 일부러 못하게 한 일

- 근거 없는 법령/판례/벤치마크 숫자 생성
- 최신 시장가격, 뉴스, 웹 검색이 필요한 답변 생성
- destructive shell command 실행
- GPU 학습 프로세스 제어
- workspace 밖 파일 읽기
- 사용자가 명시하지 않은 write 작업

## 파일 구조

```text
agent_harness/
  README.md
  agentic_eval.py
  agentic_eval_tasks.jsonl
  lfm2ko_agent.py
  profiles.json
  smoke_tasks.jsonl
scripts/
  run_lfm2ko_agent_harness.sh
  run_lfm2ko_agentic_eval.sh
  run_lfm2ko_agent_smoke_eval.sh
```

## GPU 없이 mock 테스트

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_lfm2ko_agent_smoke_eval.sh
```

이 명령은 실제 모델을 부르지 않고 profile routing, tool-call parsing, read_file
흐름만 확인한다.

에이전틱 task suite mock 흐름을 확인할 때만 다음을 실행한다. mock 답변은 실제
성능 판정용이 아니므로 실패가 있어도 종료 코드는 0으로 둔다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_lfm2ko_agentic_eval.sh
```

출력은 기본적으로 다음 JSONL에 쌓인다.

```text
/home/work/.data/lfm2_ko_sft/eval/agentic_harness/<RUN_ID>.jsonl
```

## 최종 SFT 모델로 테스트

Stage2와 최종 평가가 끝난 뒤 vLLM OpenAI-compatible endpoint를 띄운다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
CUDA_VISIBLE_DEVICES=0 \
MODEL_ID=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full \
SERVED_MODEL_NAME=lfm2-ko-sft \
PORT=1053 \
MAX_MODEL_LEN=8192 \
bash scripts/start_vllm_openai_server_for_harness.sh
```

다른 터미널에서:

```bash
OPENAI_BASE_URL=http://localhost:1053/v1 \
OPENAI_API_KEY=EMPTY \
MODEL_NAME=lfm2-ko-sft \
bash scripts/run_lfm2ko_agent_harness.sh \
  "README.md를 읽고 이 모델의 학습/평가 실행법을 한국어로 요약해라."
```

vLLM 실제 task suite:

```bash
MODE=real \
AGENT_BACKEND=vllm \
OPENAI_BASE_URL=http://localhost:1053/v1 \
OPENAI_API_KEY=EMPTY \
MODEL_NAME=lfm2-ko-sft \
EXECUTE_TOOLS=1 \
ALLOW_SHELL=1 \
bash scripts/run_lfm2ko_agentic_eval.sh
```

## GGUF / llama.cpp CPU 서버

GGUF 변환본이 생기면 같은 harness를 llama.cpp 서버에도 붙인다. llama.cpp의
`llama-server`는 OpenAI-compatible `/v1/chat/completions` endpoint를 제공하므로
harness는 backend/port만 바꾸면 된다.

서버:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
MODEL_GGUF=/path/to/LFM2.5-8B-A1B-KO-SFT-Q8_0.gguf \
MODEL_ALIAS=lfm2-ko-sft-gguf \
CTX_SIZE=8192 \
THREADS=$(nproc) \
PORT=8080 \
bash scripts/start_llamacpp_gguf_server_for_agent.sh
```

클라이언트:

```bash
AGENT_BACKEND=llamacpp \
OPENAI_BASE_URL=http://localhost:8080/v1 \
MODEL_NAME=lfm2-ko-sft-gguf \
bash scripts/run_lfm2ko_agent_harness.sh \
  --context-window 8192 \
  --prompt-budget 20000 \
  "README.md를 읽고 이 모델의 학습/평가 실행법을 한국어로 요약해라."
```

GGUF 실제 task suite:

```bash
MODE=real \
AGENT_BACKEND=llamacpp \
OPENAI_BASE_URL=http://localhost:8080/v1 \
MODEL_NAME=lfm2-ko-sft-gguf \
AGENT_CONTEXT_WINDOW=8192 \
AGENT_PROMPT_BUDGET_CHARS=20000 \
EXECUTE_TOOLS=1 \
ALLOW_SHELL=1 \
bash scripts/run_lfm2ko_agentic_eval.sh
```

컨텍스트 기준:

- vLLM 기본: `--context-window 8192`, `--prompt-budget 24000`
- GGUF CPU 기본 권장: `CTX_SIZE=8192`, `--prompt-budget 20000`
- 4k GGUF만 쓸 때: `CTX_SIZE=4096`, `--prompt-budget 10000`
- 긴 파일은 harness가 앞/뒤를 남기고 중간을 잘라 context overflow를 피한다.

## read-only tool 실행

모델이 `<|tool_call_start|>...<|tool_call_end|>` 형태로 `read_file`,
`list_files`, `shell`을 요청하면 harness가 파싱한다.

기본은 shell dry-run이다. 실제 read-only shell 실행은 명시적으로 켠다.

```bash
bash scripts/run_lfm2ko_agent_harness.sh \
  --allow-shell --execute-tools \
  "현재 git 상태와 최근 커밋 3개를 요약해라."
```

허용되는 shell prefix는 `pwd`, `ls`, `find`, `rg`, `sed`, `head`, `tail`,
`wc`, `cat`, `git status`, `git diff`, `git log`, `python -m py_compile`이다.

## 평가 후 테스트할 항목

1. 법률: 변시형 MCQA에서 `정답:` 라인과 짧은 이유를 내는지.
2. 금융: 표/수치가 주어졌을 때 계산식과 caveat를 분리하는지.
3. 터미널: 로그를 읽고 숫자 중심으로 상태를 요약하는지.
4. Text2SQL: schema가 있을 때 SQL 먼저 출력하는지.
5. 코드: 작은 수정 계획과 검증 명령을 파일 경로와 함께 쓰는지.
6. 범위 밖: 최신 가격/웹검색/전문가 조언을 지어내지 않고 제한을 말하는지.

`agent_harness/agentic_eval_tasks.jsonl`의 8개 task는 이 항목들을 빠르게 본다.
최종 full benchmark가 아니라 학습 직후 agentic 행동이 망가지지 않았는지 확인하는
gate다. 각 task는 profile 선택, tool 사용 여부, 필수 문자열, 금지 문자열을 JSONL로
남긴다. 숫자 비교가 필요한 정식 성능표는 별도 vLLM/lm-eval 결과와 합쳐 모델
카드에 올린다.

결과는 최종 모델 평가 후 README와 모델 카드의 agent behavior 섹션에 반영한다.
