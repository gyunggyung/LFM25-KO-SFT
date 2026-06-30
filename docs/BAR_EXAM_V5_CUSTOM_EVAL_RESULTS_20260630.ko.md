# 변시 v5 Custom Eval 결과 - 2026-06-30

작성 기준: 2026-06-30 15:20 KST

## 목적

공개 lm-eval 게이트와 별도로, 15회 변호사시험 문제와 v5 근거 패킷을 직접 넣었을
때 모델이 근거를 읽고 선택지 번호를 맞히는지 확인했다.

평가 대상:

| label | checkpoint |
|---|---|
| `ko_cpt` | `/home/work/.data/lfm2_ko_cpt/models/LFM2.5-8B-A1B-KO-CPT-FULL-20260628_lfm25_8b_ko_cpt_full_lfmstyle/final_full` |
| `repair_sft` | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630/final_full` |
| `bar_v5_sft` | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT-20260630/final_full` |

평가 입력:

- 문제: `/home/work/.projects/LLM-OS-Models/Terminal/Bar-exam-test/15th_split`
- 정답: `/home/work/.projects/LLM-OS-Models/Terminal/Bar-exam-test/15th_solved_v5`
- 근거: `data/bar_exam/round15_rag_contexts_v5_20260629`

## 하네스

추가한 파일:

| file | 역할 |
|---|---|
| `scripts/run_bar_exam_v5_custom_eval.py` | OpenAI-compatible endpoint에 15회 변시 v5 문제를 질의하고 정답 추출/채점 |
| `scripts/run_bar_exam_v5_custom_eval_server.sh` | 모델 하나를 vLLM으로 띄우고 custom eval 실행 |
| `scripts/run_bar_exam_v5_custom_eval_compare.sh` | KO-CPT, Repair-SFT, BarExamV5-SFT를 GPU 0/1/2에 병렬 실행 |
| `scripts/summarize_bar_exam_v5_custom_eval.py` | summary markdown 생성 |

실행 예:

```bash
LIMIT=12 \
OUT_DIR=/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_custom_20260630T_run2 \
  bash scripts/run_bar_exam_v5_custom_eval_compare.sh
```

strict 숫자 출력 평가:

```bash
LIMIT=12 MAX_TOKENS=8 STRICT_ONE_TOKEN=1 \
OUT_DIR=/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_custom_20260630T_run3_strict \
  bash scripts/run_bar_exam_v5_custom_eval_compare.sh
```

## 결과 1: 기본 `정답: N` 형식

초기 프롬프트는 `정답: N`이라는 placeholder를 그대로 복사하는 문제가 있었다.
이를 고친 뒤 다시 실행한 결과다.

경로:

```text
/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_custom_20260630T_run2/SUMMARY.md
```

| model | total | correct | accuracy | extracted | extraction_rate |
|---|---:|---:|---:|---:|---:|
| `ko_cpt` | 12 | 3 | 0.250000 | 12 | 1.000000 |
| `repair_sft` | 12 | 3 | 0.250000 | 12 | 1.000000 |
| `bar_v5_sft` | 12 | 3 | 0.250000 | 12 | 1.000000 |

대표 출력 패턴:

| model | 문제 | gold | pred | output |
|---|---|---:|---:|---|
| `bar_v5_sft` | `civil_law_01` | 3 | 2 | `정답: 2` |
| `bar_v5_sft` | `civil_law_02` | 5 | 1 | `정답: 1, 2, 3, 4, 5` |
| `bar_v5_sft` | `civil_law_05` | 1 | 1 | `정답: 1, 2` |
| `ko_cpt` | `civil_law_10` | 2 | 2 | `정답: 2, 3` |

해석:

- 세 모델 모두 12문항 중 3문항만 맞췄다.
- BarExamV5-SFT가 KO-CPT보다 높지 않았다.
- 다중 번호를 출력하는 경우가 많아서 선택지 번호를 단일 답으로 확정하는 행동이 불안정하다.

## 결과 2: strict 숫자 하나 출력

경로:

```text
/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_custom_20260630T_run3_strict/SUMMARY.md
```

| model | total | correct | accuracy | extracted | extraction_rate |
|---|---:|---:|---:|---:|---:|
| `ko_cpt` | 12 | 1 | 0.083333 | 4 | 0.333333 |
| `repair_sft` | 12 | 0 | 0.000000 | 5 | 0.416667 |
| `bar_v5_sft` | 12 | 1 | 0.083333 | 4 | 0.333333 |

대표 출력 패턴:

| model | 문제 | gold | pred | output |
|---|---|---:|---:|---|
| `bar_v5_sft` | `civil_law_01` | 3 | 3 | `정답: 3` |
| `bar_v5_sft` | `civil_law_03` | 5 | none | `ㄱ, ㄴ` |
| `bar_v5_sft` | `civil_law_05` | 1 | none | `ㄹ` |
| `ko_cpt` | `civil_law_10` | 2 | 2 | `정답: ②` |
| `repair_sft` | `civil_law_08` | 5 | 2 | `정답: ②` |

해석:

- strict로 묶으면 점수가 더 떨어진다.
- 모델들이 선택지 번호 대신 `ㄱ`, `ㄱ, ㄴ`, `ㄹ` 같은 지문 조합을 출력한다.
- 이는 “법률 근거를 읽는 능력” 이전에 “지문 판단 결과를 ①~⑤ 선택지 번호로
  매핑하는 행동”이 안정화되지 않았다는 뜻이다.

## 결론

현재 BarExamV5-SFT는 15회 v5 custom eval에서도 KO-CPT를 이기지 못했다.
`global_mmlu_full_ko_jurisprudence`의 좁은 상승은 있었지만, 실제 변시형
근거 기반 선택형 풀이에서는 아직 성공 모델로 볼 수 없다.

실패 원인은 크게 세 가지다.

1. 프롬프트가 조금만 애매하면 placeholder 또는 다중 정답을 그대로 출력한다.
2. 보기 `ㄱ/ㄴ/ㄷ/ㄹ`의 참거짓 판단과 선택지 번호 `①~⑤` 매핑이 분리되어 무너진다.
3. v5 근거를 읽더라도 최종 답을 숫자 하나로 확정하는 출력 제어가 약하다.

## 다음 해결책

추가 학습을 한다면 큰 SFT가 아니라 별도 BarExam adapter 또는 micro-SFT가 맞다.

권장 데이터 형식:

```text
문제 + v5 핵심 근거
-> 지문 O/X 표
-> 목표 지문 집합
-> 선택지 번호 대조
-> 정답: N
```

그리고 같은 문제에 대해 다음 짧은 answer-only variant를 많이 넣어야 한다.

```text
입력: 문제 + 보기 + 목표 지문 집합
출력: 정답: N
```

학습 전에 먼저 해야 할 일은 전체 150문항 custom eval을 안정적으로 돌리는 것이다.
현재 12문항 smoke 결과만으로는 모델 간 유의미한 우열을 주장할 수 없고,
BarExamV5-SFT가 좋다는 근거도 없다.
