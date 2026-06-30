# 최종 정리 - 2026-06-30

작성 기준: 2026-06-30

## 최종 결론

현재 전반 성능을 바로 올릴 확실한 후속 학습 방법은 없다. 대표 모델은
`LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`로 두고, KO-SFT 계열은 실패/진단
실험으로 정리한다.

핵심 판단:

- KO-CPT는 현재 가장 강한 공개 벤치마크 기준점이다.
- Stage2 KO-SFT는 대부분의 공개 벤치마크에서 KO-CPT보다 낮다.
- Stage3 Agentic/Fable은 행동 실험으로는 의미가 있지만 공개 벤치마크 repair는 아니다.
- Repair-SFT는 CPT 성능 회복에 실패했다.
- BarExamV5-SFT는 법학 jurisprudence 일부만 좋아졌고, 변시 custom eval과 broad MCQA에서는 실패했다.
- 지금 더 학습하면 성능을 회복하기보다 출력 분포를 더 망칠 위험이 크다.

최종 교훈은 명확하다. CPT는 한국어 지식과 도메인 지식을 늘리고 일부 전반
벤치마크를 올릴 수 있지만, 그 과정에서 다지선다, 짧은 정확답, 선택지 매핑
능력을 잃을 수 있다. 일반적인 SFT는 이 손실을 쉽게 복구하지 못했고, 오히려
verbose assistant 분포로 이동하면서 MCQA/exact-answer 점수를 더 떨어뜨렸다.
따라서 지금은 SFT를 더 얹는 것이 아니라 KO-CPT를 대표 모델로 두고 실패 원인과
재현 가능한 결과를 정리하는 것이 맞다.

변호사시험도 같은 결론이다. 오픈 모델 단독 생성으로 국내 변시를 안정적으로
푸는 방식은 현재 신뢰하기 어렵다. 가능성이 있는 경로는 법령/판례/해설 근거를
컨텍스트로 넣고, 지문 O/X 판단, `ㄱ/ㄴ/ㄷ/ㄹ` 조합, `①~⑤` 번호 매핑을 별도
절차로 강제하는 grounded workflow/RAG 방식이다. SFT는 그 절차를 보조할 수는
있지만, 이번 실험 기준으로 SFT만으로 변시 풀이 능력을 안정화하지 못했다.

## 모델별 판정

| 모델 | 상태 | 판정 |
|---|---|---|
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL` | 완료/업로드 | 대표 모델 |
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT` | 완료/업로드 | 실패 실험, 재현용 |
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT` | 완료/업로드 | 행동 주입 진단용, 공개 벤치 개선 실패 |
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT` | 완료/업로드 | CPT repair 실패 |
| `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT` | 완료/업로드 | 변시 v5 진단 실패, 좁은 법학 개선만 있음 |

## 주요 성능 요약

Stage2 KO-SFT는 공개 벤치에서 KO-CPT보다 낮았다.

| task | Base | KO-CPT | KO-SFT Stage2 | 판단 |
|---|---:|---:|---:|---|
| IFEval | 0.2921 | 0.3216 | 0.1738 | 실패 |
| GSM8K | 0.4845 | 0.5701 | 0.3381 | 실패 |
| BoolQ | 0.6544 | 0.7902 | 0.6664 | Base보다 약간 높지만 CPT보다 낮음 |
| ARC-Challenge | 0.3771 | 0.4241 | 0.2287 | 실패 |
| PIQA | 0.7203 | 0.7476 | 0.5930 | 실패 |
| KMMLU direct hard | 0.2015 | 0.1720 | 0.1055 | 실패 |
| MMLU-ProX Lite KO | 0.2585 | 0.1667 | 0.0867 | 실패 |

Repair/BarExamV5 후속 실험도 CPT를 회복하지 못했다.

| task | KO-CPT | Repair-SFT | BarExamV5-SFT | 판단 |
|---|---:|---:|---:|---|
| BoolQ | 0.7902 | 0.663303 | 0.658716 | CPT보다 낮음 |
| ARC-Challenge | 0.4241 | 0.211604 | 0.209898 | CPT보다 낮음 |
| GSM8K | 0.5701 | 0.329795 | 미실행 | repair 실패 |
| IFEval | 0.3216 | 0.181146 | 0.188540 | CPT보다 낮음 |
| Global MMLU KO jurisprudence | 0.2685 | 0.250000 | 0.296296 | V5의 좁은 개선 |
| KMMLU direct hard | 0.1720 | 0.102339 | 0.101608 | 실패 |
| MMLU-ProX Lite KO | 0.1667 | 0.091837 | 0.068027 | 실패 |

변시 v5 custom smoke eval도 성공하지 못했다.

| mode | KO-CPT | Repair-SFT | BarExamV5-SFT |
|---|---:|---:|---:|
| 기본 `정답: N` 형식, 12문항 | 3/12 | 3/12 | 3/12 |
| strict 숫자 하나 출력, 12문항 | 1/12 | 0/12 | 1/12 |

## 실패 원인

첫째, SFT 데이터 목표와 공개 벤치마크 채점 방식이 달랐다. 데이터는 긴 한국어
법률/금융 답변, 터미널/도구 사용, 코딩, 근거형 QA를 많이 포함했다. 그러나
공개 MCQA는 짧은 선택지 확률 또는 정확한 최종 답 추출을 본다. 모델이 더
친절하고 길게 말하게 되면 공개 MCQA 점수는 내려갈 수 있다.

둘째, response-only chat SFT가 정답 선택 분포를 직접 최적화하지 않았다. 특히
KMMLU, MMLU-ProX, Global MMLU KO류는 “보기 중 하나를 고르는 행동” 자체가
중요한데, 기존 SFT는 이 행동을 충분히 보강하지 못했다.

셋째, BarExamV5는 법률 근거 풀이 스타일을 주입했지만, `ㄱ/ㄴ/ㄷ/ㄹ` 판단 결과를
`①~⑤` 선택지 번호로 매핑하는 행동이 안정화되지 않았다. 실제 출력에서도
`ㄱ`, `ㄱ, ㄴ`, `정답: 1, 2` 같은 형태가 반복됐다.

넷째, 큰 full-parameter SFT는 KO-CPT의 좋은 분포를 쉽게 망친다. Repair-SFT의
131.6M tokens도 “정답 형식만 살짝 고치는” 목적에는 컸고, 1e-6 learning rate도
보수적 repair로는 높았을 가능성이 있다.

## 당장 하지 않을 일

- 추가 GPU 학습을 하지 않는다.
- 실패한 KO-SFT/Agentic/Repair/V5 checkpoint에서 이어서 학습하지 않는다.
- BarExamV5-SFT를 대표 모델처럼 홍보하지 않는다.
- 작은 smoke 결과를 성공처럼 말하지 않는다.
- CPT보다 낮은 SFT 점수를 숨기지 않는다.

## 나중에 다시 한다면

재시도는 KO-CPT에서 시작해야 한다. 목표는 “더 많이 학습”이 아니라 “CPT의 좋은
분포를 거의 건드리지 않고 정답 형식과 다지선다 선택 행동만 고치는 것”이다.

권장 방향:

| 항목 | 권장 |
|---|---|
| 시작점 | KO-CPT |
| 데이터량 | 5M-20M tokens부터 |
| 학습률 | 2e-7 ~ 5e-7 |
| 방식 | LoRA/adapter 또는 매우 짧은 full SFT |
| 데이터 | answer-only MCQA, 선택지 번호 매핑, 짧은 rationale, train-only evidence QA |
| 제외 | 긴 chat-only, full solution 과다, terminal/tool trace 과다, test split 의심 데이터 |
| gate | 100/300/500 step마다 BoolQ, ARC, GSM8K, IFEval, KMMLU, MMLU-ProX, 변시 custom eval |
| 중지 기준 | KO-CPT 대비 하락하면 즉시 폐기 |

변시를 다시 하려면 먼저 전체 150문항 custom evaluator를 안정화하고, 다음
형식의 micro 데이터만 별도 adapter로 학습하는 쪽이 맞다.

```text
문제 + v5 핵심 근거
-> 지문 O/X 표
-> 목표 지문 집합
-> 선택지 번호 대조
-> 정답: N
```

## 최종 메시지

이번 SFT 라인은 실패를 확인한 실험이다. 그래도 의미는 있다. KO-CPT가 현재
대표 모델이라는 기준이 명확해졌고, SFT가 왜 공개 벤치 성능을 망칠 수 있는지,
한국어 다지선다/변시형 선택지 매핑에서 무엇이 부족한지 확인했다.

마무리 기준은 다음이다.

- 대표 모델: KO-CPT
- SFT/Agentic/Repair/V5: 재현 가능한 negative result
- 다음 액션: 당장 학습 없음, 문서/카드/코드 정리 완료
