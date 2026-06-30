# 변시 v5 근거 기반 SFT 준비 계획

작성일: 2026-06-30

## 결론

이 데이터는 15회 변호사시험 정답 번호를 외우게 만드는 용도가 아니다.
목표는 “문제와 근거 패킷이 주어졌을 때 법령, 판례, 수동 검증 포인트를
읽고 선택지별 O/X 판단을 한 뒤 최종 번호를 내는 방식”을 학습시키는
것이다.

따라서 기본 실험 순서는 다음이 맞다.

1. KO-CPT를 시작점으로 둔다.
2. 다지선다/정답 형식 약점만 작은 repair SFT로 먼저 고친다.
3. 그 다음 변시 v5 근거 기반 SFT를 짧게 붙인다.
4. 실패한 Stage2 KO-SFT 또는 Agentic/Fable SFT에서 이어서 학습하지 않는다.

현재 사용자는 GPU 학습 금지를 명령했으므로, 여기서는 CPU 전처리와
가드된 실행 코드만 준비했다.

## 왜 이렇게 하는가

기존 KO-CPT는 법령, 판례, 한국어 문서 독해에는 강해졌지만 공개
다지선다 평가에서 답안 형식과 선택지 매칭이 흔들렸다. 반대로 큰 SFT는
다양한 채팅/도구/코딩 데이터를 섞으면서 공개 벤치마크 점수를 크게
깎았다. 이 문제를 다시 만들지 않으려면 큰 범용 SFT가 아니라 작은
목표형 SFT가 필요하다.

변시 v5에서 실제로 필요한 능력은 다음이다.

- 문제 방향 판별: `옳은 것`, `옳지 않은 것`, `모두 고른 것`.
- 보기 단위 분해: `ㄱ/ㄴ/ㄷ/ㄹ` 또는 `①~⑤`를 각각 판단.
- 근거 우선순위: v5 수동 검증 보강 포인트를 자동 후보보다 우선.
- 선택지 대조: 목표 지문 집합과 선택지 조합이 일치하는지 검산.
- 출력 안정화: 최종 답은 `정답: N` 또는 `최종 답: N번.`으로 제한.

소형 모델에서는 긴 자율 검색과 긴 추론을 한 번에 완벽히 기대하기
어렵다. 대신 v5처럼 근거가 정리된 패킷을 제공하고 출력 형식을 좁히면
성공 가능성이 높다.

## 데이터 구성

CPU 전처리 산출물:

```text
/home/work/.data/lfm2_ko_sft/prepared/bar_exam_v5/20260630_bar_exam_v5_context_solver_8192
```

검증된 최종 준비셋:

| 항목 | 값 |
|---|---:|
| raw JSONL rows | 6,376 |
| prepared samples | 6,374 |
| LFM tokens | 5,863,863 |
| max sequence length | 8,192 |
| validation errors | 0 |
| truncated rows | 91 |
| dedupe dropped | 2 |

카테고리별 구성:

| category | rows | 목적 |
|---|---:|---|
| `mcqa_safe_answer_symbol` | 1,888 | 1~14회 안전 정답 라벨, 원문 `①~⑤` 형식 |
| `mcqa_safe_answer_numeric` | 1,888 | 1~14회 안전 정답 라벨, 평가 친화 `1.~5.` 형식 |
| `current_law_bar_simple` | 1,000 | 현행 법령 직접 근거 문제 |
| `current_law_bar_hard` | 1,000 | 현행 법령+판례 혼합 난도 문제 |
| `v5_answer_free_procedure` | 150 | 15회 v5 근거 패킷 읽기 절차, 정답 번호 미출력 |
| `legal_search_first_action` | 150 | 법령/판례 검색 첫 행동 학습 |
| `v5_context_grounded_full_solution` | 150 | 15회 문제+v5 근거 -> 만점 풀이 전체 |
| `v5_context_grounded_answer_compact` | 150 | 15회 문제+v5 근거 -> 짧은 근거와 답 |

## 1~14회와 15회의 역할

1~14회는 다지선다 형식 보강용이다. `processed_multiple_choice/questions.csv`
에는 1~15회 전체 2,250문항이 있으나, answer 열에는 `30`, `37`, `40` 같은
오염 값이 섞여 있다. 그래서 전처리에서는 정답이 정확히 `1`~`5`로 확인되는
문항만 사용했다.

안전하게 사용한 1~14회 문항 수:

| 회차 | 안전 문항 |
|---:|---:|
| 1 | 135 |
| 2 | 135 |
| 3 | 135 |
| 4 | 135 |
| 5 | 135 |
| 6 | 135 |
| 7 | 135 |
| 8 | 135 |
| 9 | 134 |
| 10 | 135 |
| 11 | 134 |
| 12 | 135 |
| 13 | 135 |
| 14 | 135 |

15회는 고품질 v5 근거 패킷과 `Bar-exam-test/15th_solved_v5` 만점 풀이가
있으므로, 정답 번호만 외우는 샘플이 아니라 다음 입력 구조로 쓴다.

```text
15th_split 원문 문제
+ round15_rag_contexts_v5_20260629 근거 패킷
-> 15th_solved_v5 만점 풀이
```

즉 “근거가 주어지면 그 근거에 맞게 행동하는 법률 모델”을 만드는
데 쓰는 것이다. 15회를 별도 clean benchmark로 남기고 싶다면
`MODE=holdout_clean`으로 다시 전처리하면 15회 정답 라벨을 제외한다.

## 선택지 기호 처리

입력은 두 버전을 모두 만든다.

- `①~⑤`: 실제 변호사시험 원문과 v5 문서에 강하게 맞춤.
- `1.~5.`: 평가 하네스와 숫자 답안 출력에 강하게 맞춤.

출력은 반드시 숫자로 정규화한다.

```text
정답: 3
최종 답: 3번.
```

이렇게 해야 다지선다 평가에서 `③`, `3번`, `정답은 3` 같은 변형 때문에
채점이 흔들리는 문제를 줄일 수 있다.

## 전처리 실행

GPU를 쓰지 않는다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_prepare_bar_exam_v5_sft.sh
```

모드 선택:

```bash
MODE=holdout_clean bash scripts/run_prepare_bar_exam_v5_sft.sh
MODE=context_solver bash scripts/run_prepare_bar_exam_v5_sft.sh
MODE=product_tuned bash scripts/run_prepare_bar_exam_v5_sft.sh
```

권장 기본값은 `context_solver`다. 지금 목표가 15회 clean benchmark가 아니라
v5 근거를 주고 제대로 풀게 만드는 것이기 때문이다.

## 학습 실행

기본 실행은 거부된다. GPU 학습을 실수로 시작하지 않기 위함이다.

```bash
bash scripts/run_bar_exam_v5_sft_train_guarded.sh
```

나중에 명시적으로 승인된 경우에만:

```bash
ALLOW_TRAIN=1 bash scripts/run_bar_exam_v5_sft_train_guarded.sh
```

권장 시작점은 repair SFT 완료 체크포인트다.

```text
/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT-20260630/final_full
```

이 체크포인트가 없으면 스크립트는 자동으로 실패한다. 바로 KO-CPT에서
시작하려면 명시적으로 다음을 켜야 한다.

```bash
ALLOW_BASELINE_CPT_START=1 ALLOW_TRAIN=1 bash scripts/run_bar_exam_v5_sft_train_guarded.sh
```

기본 하이퍼파라미터는 보수적으로 잡았다.

| 항목 | 값 |
|---|---:|
| max seq length | 8192 |
| learning rate | 5e-7 |
| epochs | 1 |
| per-device batch | 1 |
| grad accumulation | 16 |

데이터가 5.86M 토큰뿐이라 긴 학습이 아니라 짧은 능력 주입 스테이지다.

## 기대 효과

기대하는 개선은 범용 벤치마크 상승이 아니라 다음이다.

- 변시형 지문/선택지 구조 인식.
- 정답 번호 출력 안정화.
- v5 근거 패킷을 읽고 선택지별 판단표를 만드는 능력.
- 법령/판례 근거가 있는 경우 해당 근거를 답안에 연결하는 능력.
- “옳은 것/옳지 않은 것/모두 고른 것” 방향 실수 감소.

기대하지 말아야 할 것은 다음이다.

- 검색 없이 새로운 판례를 스스로 찾아오는 능력.
- 1M 컨텍스트 모델처럼 긴 자료 전체를 완전히 자율 탐색하는 능력.
- 실패한 대형 SFT의 범용 벤치마크 하락을 이 작은 데이터만으로 모두 복구하는 것.

## 다음 검증

학습이 승인되어 끝난 뒤에는 전체 벤치 전에 작은 게이트를 먼저 본다.

1. 15회 v5 context solver 150문항.
2. 1~14회 holdout 일부 다지선다 숫자 답안.
3. current-law simple/hard 100~200문항 샘플.
4. 기존 KO-CPT와 같은 프롬프트로 비교.

이 게이트에서 KO-CPT보다 변시-v5 task는 올라가고, 공개 MCQA가 추가로
무너지지 않을 때만 더 큰 평가를 진행한다.
