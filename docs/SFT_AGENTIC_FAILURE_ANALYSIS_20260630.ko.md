# KO-SFT / Agentic 실패 분석 - 2026-06-30

작성 기준: 2026-06-30 09:07 KST

사용자 지시에 따라 추가 학습과 추가 GPU 평가는 중지했다. 이 문서는 이미 끝난
Stage2 KO-SFT와 Stage3 Agentic/Fable SFT의 데이터, 전처리, 평가 결과, 실패
원인을 정리한다.

## 결론

| 모델 | 공개 벤치 기준 결론 | 배포 상태 | 판단 |
|---|---|---|---|
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL` | 현재 가장 좋은 공개 벤치 기준 한국어 모델 라인 | 업로드 완료 | 유지 |
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT` | Stage2 SFT는 다수 공개 벤치에서 Base/CPT보다 하락 | 업로드 완료 | 실패 실험으로 기록 |
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT` | 일부 항목 소폭 회복은 있으나 공개 벤치 개선 모델은 아님 | 업로드 완료 | 실패 실험으로 기록 |

현재 홍보나 대표 모델로 내세울 기준점은 KO-CPT다. Stage2 KO-SFT와 Stage3
Agentic/Fable은 재현성과 원인 분석을 위해 공개하되, 성능 개선 모델로 주장하지
않는다.

## 학습 데이터와 전처리

### Stage2 KO-SFT

Stage2까지의 main SFT 총량은 약 4.309577B tokens다.

| stage | samples | tokens | max seq | 목적 |
|---|---:|---:|---:|---|
| Stage1 4k finance/Text2SQL | 2,302,304 | 1,285,864,494 | 4096 | 금융, 회계, Text2SQL |
| Stage1 8k legal/terminal | 1,600,835 | 1,658,848,754 | 8192 | 법률 장문, 터미널, 툴 사용 |
| Stage2 diverse plus KoTSQA | 1,468,598 | 1,364,863,776 | 4096 | 한국어 도메인, SWE, reasoning, KoTSQA train |

전처리 핵심:

- Liquid LFM chat template 기준으로 `system`, `user`, `assistant`, `tool` role을 유지했다.
- LFM tokenizer로 전부 다시 토크나이즈했다. 초기 prepared 데이터의 LFM vocab
  초과 token id 때문에 CUDA device-side assert가 났기 때문이다.
- assistant 응답에만 loss를 주는 response-only labels를 사용했다.
- KoTSQA는 `train` split만 사용했고 `test` split은 평가용으로 남겼다.
- raw CPT 성격의 대용량 말뭉치, 예를 들어 Korean Wiki/raw law/raw term corpus는
  SFT에 다시 넣지 않았다. 이미 CPT에서 지식 주입을 했기 때문이다.

### Stage3 Agentic/Fable

Agentic/Fable stage는 별도 후속 실험이다.

| item | value |
|---|---:|
| samples | 3,943 |
| tokens | 7,124,298 |
| max seq | 8192 |
| 목적 | 터미널, 로그 진단, 문서 근거 추론, 안전한 명령 계획 |

사용 데이터:

- `fable_distillation/datasets_ko/fable5_ko_sft_20260624.jsonl`
- `fable_distillation/datasets_ko/helio_ko_sft_20260628.jsonl`
- 이 워크스페이스의 README, runbook, train logs, git/vLLM 오류 패턴으로 만든
  local grounded examples

전처리는 Stage2와 동일하게 LFM chat template과 response-only labels를 사용했다.

## Stage2 공개 벤치 결과

Base/CPT 값은 CPT model card 기준값이고, KO-SFT Stage2는 업로드된 Stage2
checkpoint를 vLLM/lm-eval로 평가한 값이다.

| task | metric | Base | CPT | KO-SFT Stage2 | SFT vs Base | SFT vs CPT |
|---|---|---:|---:|---:|---:|---:|
| IFEval | prompt loose acc | 0.2921 | 0.3216 | 0.1738 | -0.1183 | -0.1478 |
| Leaderboard IFEval | prompt loose acc | 0.2902 | 0.3457 | 0.1756 | -0.1146 | -0.1701 |
| GSM8K | exact match | 0.4845 | 0.5701 | 0.3381 | -0.1464 | -0.2320 |
| BoolQ | acc | 0.6544 | 0.7902 | 0.6664 | +0.0120 | -0.1238 |
| ARC-Challenge | acc_norm | 0.3771 | 0.4241 | 0.2287 | -0.1484 | -0.1954 |
| PIQA | acc_norm | 0.7203 | 0.7476 | 0.5930 | -0.1273 | -0.1546 |
| Global MMLU KO medical genetics | acc | 0.2900 | 0.3800 | 0.3000 | +0.0100 | -0.0800 |
| Global MMLU KO nutrition | acc | 0.2549 | 0.3203 | 0.2157 | -0.0392 | -0.1046 |
| Global MMLU KO philosophy | acc | 0.2669 | 0.3215 | 0.1994 | -0.0675 | -0.1221 |
| Global MMLU KO miscellaneous | acc | 0.3372 | 0.3921 | 0.2401 | -0.0971 | -0.1520 |
| Global MMLU KO professional medicine | acc | 0.3235 | 0.2316 | 0.1838 | -0.1397 | -0.0478 |
| Global MMLU KO high school statistics | acc | 0.2870 | 0.1574 | 0.2222 | -0.0648 | +0.0648 |
| Global MMLU KO astronomy | acc | 0.3421 | 0.2829 | 0.1974 | -0.1447 | -0.0855 |
| Global MMLU KO high school computer science | acc | 0.3100 | 0.2800 | 0.2800 | -0.0300 | +0.0000 |
| Global MMLU KO jurisprudence | acc | 0.2870 | 0.2685 | 0.2593 | -0.0277 | -0.0092 |
| KMMLU direct hard | exact match | 0.2015 | 0.1720 | 0.1055 | -0.0960 | -0.0665 |
| KMMLU direct hard STEM | exact match | 0.1973 | 0.1564 | 0.0773 | -0.1200 | -0.0791 |
| MMLU-ProX Lite KO | exact match | 0.2585 | 0.1667 | 0.0867 | -0.1718 | -0.0800 |

좋게 나온 항목은 제한적이다.

- BoolQ는 Base보다 +0.0120이지만 CPT보다 -0.1238이다.
- Global MMLU KO medical genetics는 Base보다 +0.0100이지만 CPT보다 -0.0800이다.
- Global MMLU KO high school statistics는 CPT보다 +0.0648 회복했지만 Base보다
  -0.0648이다.
- high school computer science는 CPT와 동률이지만 Base보다 낮다.

따라서 Stage2는 “일부 회복 항목은 있으나 전체적으로 실패”로 판정한다.

## Stage3 Agentic/Fable 결과

Agentic/Fable은 Stage2 대비 일부 항목을 소폭 회복했지만, Base/CPT를 이기는
공개 벤치 모델이 아니다.

| task | Stage2 | Agentic/Fable | 변화 | 해석 |
|---|---:|---:|---:|---|
| Global MMLU KO limit50 | 0.244681 | 0.251773 | +0.007092 | 소폭 회복 |
| Global MMLU KO medical limit50 | 0.361111 | 0.416667 | +0.055556 | 작은 샘플에서 회복 |
| IFEval strict limit50 | 0.1000 | 0.1000 | +0.0000 | 개선 없음 |
| KMMLU direct hard limit50 | 0.113407 | 0.109734 | -0.003673 | 하락 |
| MMLU-Pro law | 0.134423 | 0.150772 | +0.016349 | 소폭 회복 |
| MMLU-Pro economics | 0.323460 | 0.331754 | +0.008294 | 소폭 회복 |
| TruthfulQA MC2 | 0.474975 | 0.476824 | +0.001849 | 거의 동일 |
| BoolQ | 0.6664 | 0.664220 | -0.002180 | 하락 |
| GSM8K exact | 0.3381 | 0.360879 | +0.022779 | 회복했지만 CPT/Base보다 낮음 |
| PIQA acc_norm/acc | 0.5930 | 0.588139 acc | N/A | metric 차이 주의, 여전히 낮음 |

Agentic/Fable stage는 7.12M tokens로 매우 작다. 이 정도 규모는 터미널/로그
습관을 주입하는 데는 의미가 있을 수 있지만, 4.3B-token SFT 이후 망가진 공개
벤치 성능을 복구하기에는 충분하지 않았다.

## 실패 원인

### 1. SFT 목적과 공개 벤치 측정 방식이 다르다

Stage2 데이터는 한국어 법률/금융/터미널/코딩/근거형 QA를 많이 포함한다.
이 데이터는 긴 설명, 절차형 답변, JSON/SQL/터미널 형태의 assistant 응답을
강화한다. 반면 KMMLU, Global MMLU KO, MMLU-ProX는 선택지의 loglikelihood나
짧은 exact-answer를 보는 경우가 많다.

결과적으로 모델은 “자연어로 길게 답하는 행동”을 더 배우고, “보기 A/B/C/D 중
정답 토큰만 강하게 고르는 행동”은 약해졌다.

### 2. response-only SFT가 선택지 scoring을 직접 보강하지 못했다

assistant 응답만 학습하는 방식은 chat 품질에는 맞지만, multiple-choice
likelihood 평가에는 간접적이다. 정답 옵션 하나만 출력하는 예시, 보기별 비교,
짧은 final-answer 고정 포맷이 충분하지 않으면 선택지 확률이 나빠질 수 있다.

### 3. CPT가 만든 지식/다지선다 균형을 SFT가 덮었다

KO-CPT는 한국어 지식과 일부 일반 benchmark를 끌어올렸다. Stage2 SFT는 이 위에
4.3B tokens를 더 학습했지만, 데이터 분포가 공개 벤치의 선택지형 형식과 다르다.
그래서 CPT의 좋은 확률 분포를 보존하기보다, chat/domain response 분포로 이동했다.

### 4. KoTSQA는 필요한 데이터지만 MCQA 수리 데이터는 아니다

KoTSQA는 근거 QA와 false-premise correction에는 좋다. 그러나 KMMLU/MMLU-ProX
형식의 다지선다 repair에는 직접적이지 않다. KoTSQA 추가만으로 한국어 다지선다
선택 능력이 회복될 것으로 보면 안 된다.

### 5. Agentic/Fable은 공개 벤치 회복 stage가 아니다

Agentic/Fable 데이터는 터미널 실행, 로그 읽기, 문서 근거 추론, 안전한 명령
계획에 맞춘 데이터다. 공개 지식/수학/다지선다 성능을 올리는 데이터가 아니며,
토큰 수도 7.12M으로 작다. 따라서 benchmark repair가 아니라 behavior injection
실험으로 봐야 한다.

### 6. 평가/런처 문제도 있었지만, 점수 하락의 주원인은 아니다

vLLM OpenAI server 쪽에서 user-site torch를 잘못 잡아 ABI 오류가 나는 문제가
있었다. 이를 막기 위해 `PYTHONNOUSERSITE=1`, `PYTHONPATH=""`를 넣었다. 또한
8 GPU eval queue가 한 작업 실패 시 전체 queue를 끊는 문제가 있어 child failure
handling과 `GPU_IDS` 지원을 넣었다.

이 문제들은 평가 안정성 문제다. Stage2/Agentic의 공개 벤치 하락 자체는 데이터
mix와 SFT 목적 불일치가 더 큰 원인이다.

## 지금 하지 말아야 할 것

- Stage2 SFT나 Stage3 Agentic을 성공 모델처럼 홍보하지 않는다.
- KO-CPT보다 좋다고 주장하지 않는다.
- 추가 SFT를 즉시 이어서 돌리지 않는다. 사용자가 추가 학습 중지를 지시했다.
- benchmark test split을 학습 데이터에 넣지 않는다.

## 다음 실험 제안

현재는 실행하지 않는다. 나중에 재개한다면 Stage2 결과 위가 아니라 KO-CPT 위에서
작은 MCQA repair SFT를 새로 하는 편이 낫다.

권장 방향:

1. KO-CPT를 시작점으로 사용한다.
2. 100M-300M tokens 규모의 작고 선별된 repair set만 사용한다.
3. 데이터는 한국어 다지선다, answer-only, 짧은 rationale, final option 분리를
   중심으로 만든다.
4. legal/bar JSON, KoTSQA train, 한국어 금융/회계 문제는 평가셋 오염을 확인한
   뒤 train split만 사용한다.
5. 매 stage 후 IFEval, GSM8K, BoolQ, Global MMLU KO, KMMLU direct hard,
   MMLU-ProX Lite KO를 짧게 gate한다.
6. gate에서 CPT보다 낮으면 즉시 중지한다.

Agentic/Fable은 공개 벤치 모델이 아니라 별도 behavior model로 분리해야 한다.
터미널 실행, 로그 진단, 근거 문서 읽기, 안전한 shell 계획을 평가하는 harness가
먼저 안정화된 뒤에만 추가 학습을 고려한다.

## 공개 링크

- KO-CPT: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL>
- KO-SFT: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT>
- KO-Agentic-SFT: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT>
- SFT GitHub: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT GitHub: <https://github.com/gyunggyung/LFM25-KO-CPT>
- Dataset releases: `docs/HF_DATASETS_20260629.ko.md`
