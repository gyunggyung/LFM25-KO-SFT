# Repair / BarExamV5 SFT 결과 분석 - 2026-06-30

작성 기준: 2026-06-30 13:40 KST

## 결론

두 후속 SFT는 모두 학습과 Hugging Face 업로드가 끝났다. 그러나 공개 게이트
평가 기준으로는 실패다. 현재 대표 모델은 여전히
`LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`로 두는 것이 맞다.

| 모델 | HF repo | 판정 |
|---|---|---|
| Repair-SFT | `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT` | CPT 성능 회복 실패 |
| Repair-BarExamV5-SFT | `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT` | 법학 일부만 개선, broad MCQA는 더 악화 |

## 학습 산출물

| 항목 | Repair-SFT | BarExamV5-SFT |
|---|---:|---:|
| 시작 checkpoint | KO-CPT final | Repair-SFT final |
| local final | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630/final_full` | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT-20260630/final_full` |
| prepared dataset | `repair_cpt/20260630_cpt_mcqa_repair_4k/lfm_chat_4k` | `bar_exam_v5/20260630_bar_exam_v5_context_solver_8192/lfm_chat_8192` |
| samples | 188,493 | 6,374 |
| LFM tokens | 131,607,379 | 5,863,863 |
| max sequence | 4,096 | 8,192 |
| learning rate | 1e-6 | 5e-7 |
| epochs | 1 | 1 |
| planned steps | 1,473 | 57 |
| upload | 완료 | 완료 |

## 게이트 평가 결과

평가 경로:

```text
/home/work/.data/lfm2_ko_sft/eval/repair_sft_gate_20260630T1306KST/SUMMARY.md
/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_sft_gate_20260630T1306KST/SUMMARY.md
```

Base/CPT 기준값은 `configs/benchmark_reference_cpt_card_20260630.json`에 기록한
CPT 모델 카드 수치다. 일부 항목은 metric이 다르다. 예를 들어 CPT card는
`acc_norm`, `flexible-extract`, `prompt_level_loose_acc`를 쓰고, 이번 gate는
`acc`, `strict-match`, `prompt_level_strict_acc`를 쓴 항목이 있다. 따라서 metric이
다른 행은 방향성 판단용으로만 본다.

| task | Base | KO-CPT | Repair-SFT | BarExamV5-SFT | 해석 |
|---|---:|---:|---:|---:|---|
| BoolQ | 0.6544 | 0.7902 | 0.663303 | 0.658716 | Repair는 Base보다 약간 높지만 CPT보다 크게 낮음 |
| ARC-Challenge | 0.3771 | 0.4241 | 0.211604 | 0.209898 | metric 차이를 감안해도 크게 낮음 |
| GSM8K | 0.4845 | 0.5701 | 0.329795 | 미실행 | CPT 회복 실패 |
| IFEval | 0.2921 | 0.3216 | 0.181146 | 0.188540 | V5가 Repair보다 조금 낫지만 CPT보다 낮음 |
| Global MMLU KO jurisprudence | 0.2870 | 0.2685 | 0.250000 | 0.296296 | V5의 유일한 명확한 개선 신호 |
| KMMLU direct hard | 0.2015 | 0.1720 | 0.102339 | 0.101608 | 한국어 다지선다 회복 실패 |
| KMMLU direct hard law | n/a | n/a | 0.170000 | 0.190000 | 법 카테고리는 소폭 개선 |
| MMLU-ProX Lite KO | 0.2585 | 0.1667 | 0.091837 | 0.068027 | V5에서 추가 악화 |
| MMLU-ProX Lite KO law | n/a | n/a | 0.104167 | 0.020833 | 법률 exact-extraction이 크게 악화 |

## 원인 판단

첫째, 이번 문제는 한국어 지식 부족보다 출력 분포 붕괴에 가깝다. KO-CPT는 이미
법령, 판례, 한국어 문서 독해를 많이 학습했다. 그런데 SFT를 얹으면 모델이
짧은 정답 선택 대신 긴 assistant 답변을 하도록 이동하면서 exact-match와
다지선다 채점이 무너진다.

둘째, Repair-SFT의 131.6M tokens도 작다고 보기 어렵다. 목표는 “CPT 분포를
거의 건드리지 않고 정답 형식만 교정”이었는데, 1e-6 learning rate와 1 epoch
full-parameter SFT는 public benchmark 분포를 다시 밀어낼 만큼 컸다.

셋째, BarExamV5 데이터는 목적 자체가 public MCQA repair가 아니다. 긴 근거
패킷, 선택지별 판단, 만점 풀이 스타일을 학습시키면 `global_mmlu_full_ko_jurisprudence`
같은 법학 지식 항목은 좋아질 수 있다. 하지만 MMLU-ProX/KMMLU처럼 짧고 정확한
선택지 추출을 요구하는 평가는 오히려 나빠질 수 있다.

넷째, metric 차이와 평가 하네스 차이를 별도로 관리해야 한다. 이번 결과가
나쁜 것은 분명하지만, CPT card와 gate의 metric이 다르면 수치 차이를 그대로
홍보 표에 섞으면 안 된다. 다음부터는 같은 task, 같은 metric, 같은 decoding,
같은 extraction으로 `Base -> CPT -> 후보`를 한 번에 비교해야 한다.

다섯째, 업로드 전에 early gate가 부족했다. Repair는 100/300/500 step에서
작은 평가를 끊어 봤어야 했고, V5는 57 step 전체가 짧더라도 시작 전부터
별도 v5 heldout evaluator와 public MCQA 보존 gate를 같이 묶었어야 한다.

## 해결 방향

다음 대표 모델을 만들려면 실패한 SFT에서 이어가지 말고 KO-CPT에서 다시 시작한다.
CPT를 더 하는 것보다 작은 SFT를 다시 설계하는 쪽이 맞다. 이유는 현재 약점이
지식량보다 정답 형식, 선택지 매칭, 출력 추출 문제에 가깝기 때문이다.

권장 재시도:

| 단계 | 권장 |
|---|---|
| base | `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL` |
| 데이터량 | 5M-20M tokens부터 시작 |
| 학습률 | 2e-7 ~ 5e-7 |
| 학습 방식 | full SFT보다 LoRA/adapter 또는 매우 짧은 full SFT 우선 |
| 데이터 | answer-only MCQA, numeric extraction, short rationale, heldout 없는 train-only evidence QA |
| 제외 | 긴 chat-only, terminal/tool trace, verbose full solution, test split 의심 데이터 |
| early gate | 100/300/500 step마다 BoolQ, ARC, GSM8K, IFEval, KMMLU, MMLU-ProX |
| 중지 기준 | CPT 대비 큰 하락이면 즉시 폐기 |

BarExamV5류 실험은 별도 모델이나 adapter로 분리한다. 목표가 “근거를 주면 법률
근거에 맞게 푸는 모델”이면 public benchmark와 같은 모델 카드에서 성공으로
홍보하지 말고, 별도 v5 heldout 평가를 만든 뒤 그 결과만 주장해야 한다.

## 현재 홍보 기준

LinkedIn이나 모델 카드에서 내세울 수 있는 것은 KO-CPT다. 이번 Repair/V5는
다음과 같이만 말해야 한다.

- KO-CPT는 현재 주력 공개 벤치 checkpoint다.
- 대형 KO-SFT, Agentic/Fable SFT, Repair-SFT, BarExamV5-SFT는 모두 진단 실험이다.
- Repair/V5 실험은 왜 SFT가 CPT 성능을 망칠 수 있는지 보여주는 negative result다.
- V5는 법학 jurisprudence gate에서만 좁은 개선을 보였고, broad MCQA에는 적합하지 않았다.

## 후속 작업

1. 두 HF 모델 카드 상단에 위 결과를 명시한다.
2. GitHub README에서 대표 모델이 KO-CPT라는 점을 다시 표시한다.
3. 다음 학습은 GPU를 쓰기 전에 5M-20M token micro-repair 데이터와 early gate
   체인을 먼저 만든다.
4. BarExamV5는 별도 heldout custom evaluator를 먼저 만든 뒤에만 추가 학습한다.
