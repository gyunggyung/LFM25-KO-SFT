# LFM2.5 KO SFT / Agentic 단계별 의미와 목표

작성일: 2026-06-29  
폴더: `/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft`

이 문서는 `LFM2.5-8B-A1B-KO-SFT`의 단계별 목표, 왜 그 순서로 학습하는지,
컨텍스트 길이를 어떻게 정했는지, 그리고 Agentic/Fable 후속 학습이 어떤 효과를
노리는지 정리한다.

## CPT 이후 SFT가 필요한 이유

CPT는 한국어 지식과 도메인 분포를 이식했다. 하지만 CPT는 next-token objective라서
다음 행동을 직접 보장하지 않는다.

- 객관식에서 선택지 문자를 정확히 출력하는 능력
- 법률/금융 답변에서 근거와 한계를 분리하는 습관
- Text2SQL처럼 출력 형식이 중요한 작업
- LFM tool-call 토큰을 정확히 쓰는 능력
- terminal/log를 읽고 안전한 명령 계획을 세우는 능력

이런 것은 raw CPT보다 response-only SFT가 더 직접적으로 고친다. 그래서 이 폴더의
핵심은 "한국어 지식이 들어간 CPT를 한국어로 잘 지시 따르는 모델로 바꾸는 것"이다.

## 공통 SFT 원칙

| 항목 | 원칙 |
|---|---|
| 방식 | full-parameter SFT, LoRA 아님 |
| label | assistant response-only labels |
| tokenizer | LFM tokenizer 고정 |
| role | `system`, `user`, `assistant`, `tool` 유지 |
| tool-call | `<|tool_call_start|>`, `<|tool_call_end|>` 보존 |
| trainer | direct `torchrun` DDP 경로 |
| checkpoint | main stages는 1000 steps, keep 2 중심 |

왜 response-only인가:

- user/system/tool context까지 loss를 주면 프롬프트를 그대로 외우는 방향으로
  불필요한 압력이 생긴다.
- SFT 목표는 "질문을 보고 답을 생성하는 행동"이므로 assistant 응답에 loss를
  집중하는 편이 맞다.

왜 full-parameter인가:

- CPT로 바뀐 내부 분포 위에서 한국어 행동을 강하게 조정해야 한다.
- LoRA는 빠르고 싸지만, 이번 목표는 공개 main 모델 자체를 만들고 다음 stage를
  이어가는 것이다.
- 법률/금융/툴콜/코딩/추론이 서로 섞여 있으므로 adapter만으로 행동을 충분히
  고정하기 어렵다고 판단했다.

## Stage0: Legal 8k Warmup

| 항목 | 값 |
|---|---:|
| samples | 8,747 |
| tokens | 35,068,923 |
| max seq | 8192 |
| 목적 | LFM tokenizer/response-only label/full SFT 저장 검증 |

데이터:

- legal source-grounded SFT
- legal RAG round15
- current law bar JSON answer SFT

왜 필요한가:

- 첫 full SFT를 바로 수B token으로 시작하면 tokenizer mismatch나 label 오류를
  늦게 발견한다.
- 실제로 이전 prepared 데이터에는 LFM vocab 범위를 넘는 token id 문제가 있었고,
  CUDA device-side assert를 낼 수 있었다.
- Stage0은 작은 법률 데이터로 8k context, LFM tokenizer, response-only labels,
  checkpoint/HF push가 모두 정상인지 확인하는 안전 단계다.

기대 효과:

- 법률 근거형 답변과 bar-style JSON 형식의 초기 행동을 넣는다.
- 후속 Stage0b/Stage1에서 학습 체인을 믿고 크게 돌릴 수 있게 한다.

## Stage0b: Finance/Text2SQL 4k Smoke

| 항목 | 값 |
|---|---:|
| samples | 280,000 |
| tokens | 58,090,087 |
| max seq | 4096 |
| planned steps | 2,188 |
| effective batch | 128 sequences/update |

데이터:

- finance BCAI 120k
- Text2SQL 160k

왜 필요한가:

- Stage0은 너무 작아서 8 GPU long run 특성을 보기 어렵다.
- Stage0b는 짧은 4k 시퀀스로 full DDP, fused AdamW, save/push 경로를 검증한다.
- finance/Text2SQL은 형식과 전문 용어가 강하므로 짧은 smoke stage에서도
  학습 신호가 명확하다.

기대 효과:

- 한국어 금융/회계 설명의 기본 언어를 안정화한다.
- Text2SQL/structured output의 정확한 응답 형식을 초기에 넣는다.

## Stage1 4k: Finance/Text2SQL Main

| 항목 | 값 |
|---|---:|
| samples | 2,302,304 |
| tokens | 1,285,864,494 |
| max seq | 4096 |
| effective batch | 128 sequences/update |

데이터:

- finance/accounting: 약 1.188B tokens
- Text2SQL: 약 0.097B tokens

왜 4k인가:

- finance/accounting와 Text2SQL 샘플은 대부분 4k 안에 들어간다.
- 4k는 per-device batch 2를 유지할 수 있어 처리량이 좋다.
- Stage1 8k보다 먼저 돌려, 비교적 빠르게 domain SFT 효과와 training stability를
  확보한다.

기대 효과:

- 한국어 금융/회계 질의응답 강화
- SQL/표/스키마 기반 구조화 응답 강화
- Stage1 8k로 넘어가기 전 main SFT checkpoint 품질 확보

## Stage1 8k: Legal/Terminal

| 항목 | 값 |
|---|---:|
| samples | 1,600,835 |
| tokens | 1,658,848,754 |
| max seq | 8192 |
| effective batch | 128 sequences/update |

데이터:

- Korean legal tasks: 약 1.056B tokens
- terminal/toolbench: 약 0.603B tokens

왜 8k인가:

- 법률 문서와 terminal/tool traces는 instruction, evidence, command/output이 함께
  들어가 4k로 자르면 핵심 근거가 잘린다.
- terminal/toolbench는 tool-call의 앞뒤 문맥이 중요하다. command만 남기면 왜
  그 명령을 선택했는지 학습하지 못한다.
- 8k는 H200 8장 기준 여전히 실용적인 full SFT 길이다.

기대 효과:

- 한국어 법률 장문 QA와 근거 중심 답변 보강
- terminal 상태 요약, 로그 읽기, 안전한 command 계획, tool-call 형식 보존
- Stage3 agentic 학습의 기반 제공

## Stage2: Diverse KO/SWE/Reasoning + KoTSQA

| 항목 | 값 |
|---|---:|
| samples | 1,468,598 |
| tokens | 1,364,863,776 |
| max seq | 4096 |

데이터:

- Korean domain core
- behavior core
- SWE/coding: SWE Zero, GLM/SWE mix, compact SWE
- reasoning/agent: GLM reasoning, HF extra reasoning/agent, compact agent reasoning
- compact finance/legal
- Text2SQL DuckDB
- KoTSQA v2 train split only

왜 Stage2가 필요한가:

- Stage1은 finance/legal/terminal 비중이 높아 특정 도메인으로 기울 수 있다.
- Stage2는 한국어 일반, 코딩, reasoning, compact domain reinforcement를 섞어
  모델을 다시 넓힌다.
- KoTSQA train은 한국어 evidence QA, false-premise correction, 표/문서 기반
  답변을 보강한다.

왜 KoTSQA test는 제외하는가:

- 후속 평가에 써야 하므로 train split만 학습한다.
- test를 학습에 넣으면 한국어 evidence QA 평가 설득력이 떨어진다.

기대 효과:

- Stage1에서 생길 수 있는 도메인 편향 완화
- 코딩/SWE와 general reasoning 복구
- 한국어 근거 기반 답변 강화
- final main SFT 모델로 공개할 균형점 확보

## Stage3: Agentic/Fable Separate SFT

| 항목 | 값 |
|---|---:|
| target repo | `LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT` |
| samples | 3,943 |
| tokens | 7,124,298 |
| max seq | 8192 |
| effective batch | 128 sequences/update |
| learning rate | `1e-6` |

데이터:

- Fable5 Korean traces
- Helio Korean reasoning traces
- local README/runbook/train log/git/vLLM/agent harness grounded examples

왜 main SFT와 분리하는가:

- Agentic behavior는 terminal/tool/log/doc diagnosis에 강한 특수 행동이다.
- main SFT 모델은 한국어 범용 법률/금융/코딩/reasoning 모델로 남겨야 한다.
- Agentic/Fable stage를 같은 repo에 덮으면, 범용 benchmark와 agent benchmark의
  변화 원인을 분리하기 어렵다.

왜 Agentic SFT를 하는가:

- CPT/SFT만으로 "도구가 필요하면 정확한 tool-call을 내고, 결과를 받은 뒤
  근거 기반으로 답하는" 루프가 자동으로 안정되지는 않는다.
- Fable-style 데이터는 명령을 바로 실행하는 것이 아니라, 상태 확인, 파일 읽기,
  로그 해석, 오류 원인 추정, 안전한 다음 명령 제안 같은 agentic 절차를 가르친다.
- local grounded examples는 이 프로젝트의 실제 문서와 로그를 넣어, 모델이 없는
  결과를 지어내지 않고 "문서에 있는 것/없는 것"을 구분하게 만드는 역할을 한다.

기대 효과:

- README/runbook/log를 읽고 현재 상태를 요약하는 능력
- tool-call syntax 정확도 상승
- shell command를 무작정 실행하지 않고 안전한 read-only 진단부터 하는 습관
- git push/reject, CUDA/OOM, vLLM server, training log 같은 실제 운영 오류 설명
- Codex/Claude Code류 작업 흐름에 가까운 "확인 -> 판단 -> 실행 계획 -> 검증"
  형태의 답변

리스크:

- 너무 많은 agentic trace를 넣으면 일반 QA 답변이 장황해질 수 있다.
- tool-call이 필요 없는 질문에서도 도구를 부르려는 과잉 행동이 생길 수 있다.
- 그래서 Stage3는 별도 repo로 두고, main SFT와 비교 평가한다.

## Context Length 정책

| 단계 | context | 이유 |
|---|---:|---|
| CPT | 8192 | 법률/위키/terminal long text를 가능한 한 유지 |
| Stage0 legal | 8192 | legal source/RAG/bar JSON 검증 |
| Stage0b finance/Text2SQL | 4096 | 짧은 smoke stage와 처리량 확보 |
| Stage1 finance/Text2SQL | 4096 | 대부분 4k 안에 들어가고 batch 효율이 좋음 |
| Stage1 legal/terminal | 8192 | 장문 법률/terminal trace 보존 |
| Stage2 diverse/KoTSQA | 4096 | 일반 SFT/SWE/reasoning 균형과 속도 |
| Stage3 Agentic/Fable | 8192 | docs/log/tool-call context 보존 |
| GGUF/CPU harness | 8192 권장 | llama.cpp에서도 현실적인 기본값 |

16k/32k는 후속 실험이다. 지금 체인은 8192를 기준으로 학습/평가/하네스가 맞춰져
있고, 6월 30일 마감 전에는 이 기준을 깨지 않는 것이 안전하다.

## 평가 원칙

Stage2 이후:

- base, CPT, SFT를 같은 vLLM/lm-eval 조건으로 비교한다.
- Korean hard MCQA가 계속 하락하는지 확인한다.
- IFEval/GSM8K/BoolQ/ARC 같은 보존 항목도 같이 본다.

Agentic 이후:

- main Stage2와 Agentic Stage3를 비교한다.
- 일반 benchmark가 심하게 무너졌는지 확인한다.
- 별도 agent harness로 다음을 본다:
  - tool-call syntax
  - README/runbook/log grounded answer
  - command safety
  - code/test plan
  - legal/finance 답변의 근거와 한계

## 작업 순서의 의미

1. CPT: 한국어 지식과 도메인 분포를 넣는다.
2. Stage0/0b: LFM SFT 포맷과 full DDP 체인을 검증한다.
3. Stage1: finance/legal/terminal 핵심 타깃을 크게 학습한다.
4. Stage2: general Korean, SWE, reasoning, KoTSQA로 균형을 맞춘다.
5. Stage3 Agentic: 별도 모델로 tool/log/doc grounded agent behavior를 넣는다.
6. 평가/모델카드/HF dataset 공개: 어떤 능력이 올랐고 내려갔는지 숫자로 남긴다.

이 순서를 지키는 이유는 GPU 시간을 낭비하지 않으면서도, 실패했을 때 어느 단계가
원인인지 분리하기 위해서다.
