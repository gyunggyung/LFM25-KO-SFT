# KO-CPT 보존형 Repair SFT 계획 - 2026-06-30

작성 기준: 2026-06-30 09:46 KST

현재 사용자 지시에 따라 학습과 GPU 평가는 하지 않는다. 이 문서는 다음 실험을
재개할 때 필요한 데이터 준비, 검증 루프, 학습 guard, 예상 시간을 정리한다.

## 결론

다음 실험은 실패한 KO-SFT/Agentic checkpoint에서 이어가지 말고
`LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`에서 다시 시작하는 작은 repair SFT가
맞다.

| 선택지 | 판단 | 이유 |
|---|---|---|
| CPT를 더 한다 | 후순위 | 현재 실패 원인은 지식 부족보다 answer format/MCQA scoring 붕괴에 가까움 |
| 실패한 SFT에서 이어서 SFT | 비권장 | 이미 망가진 선택지 확률분포 위에서 원인 분리가 어려움 |
| KO-CPT에서 작은 repair SFT | 권장 | CPT의 좋은 public benchmark 분포를 시작점으로 보존하면서 약점만 고침 |

이번 실패는 “한국어 지식이 부족해서”라기보다 “긴 chat/domain SFT가 선택지형
평가와 정확답 형식을 덮어서” 발생했다. 따라서 새 CPT로 raw corpus를 더 넣기보다,
KO-CPT 위에 100M-300M tokens의 작은 answer-format repair SFT를 먼저 해야 한다.

## 목표

1. KO-CPT의 공개 벤치 강점을 보존한다.
2. Stage2 SFT에서 망가진 IFEval, GSM8K, ARC, PIQA, KMMLU, MMLU-ProX Lite KO를
   gate로 감시한다.
3. 한국어 다지선다와 정확답 포맷을 직접 보강한다.
4. 장문 chat-only 데이터와 Agentic/Fable trace를 main repair에 섞지 않는다.

## 데이터 구성

설정 파일: `configs/cpt_repair_sft_sources_20260630.json`

권장 full preprocess 목표는 200M tokens였다. 실제 compact source를 모두 사용한
결과 131.6M LFM tokens가 만들어졌다. 이는 후보 범위 100M-300M tokens 안에 있다.
억지로 200M을 채우려면 더 큰 chat/terminal 데이터를 넣어야 하는데, 이번 실패
원인이 chat/domain SFT drift였으므로 1차 repair 후보는 131.6M으로 유지한다.

| category | target ratio | source | 이유 |
|---|---:|---|---|
| Korean MCQA exact-answer | 35% | current law bar JSON, answer-only variants, compact JSON variants, short rationale variants | KMMLU/MMLU-ProX/exact-answer 붕괴를 직접 수리 |
| Korean evidence QA | 15% | KoTSQA v2 train only | 근거 QA/false-premise 보존, test split 오염 방지 |
| Finance/Text2SQL exact | 15% | compact finance, DuckDB Text2SQL | 도메인/구조화 출력 유지 |
| Coding/SWE preservation | 15% | compact SWE Zero | 코딩/SWE 능력 보존 |
| Reasoning/instruction preservation | 20% | compact GLM reasoning, compact agent reasoning, behavior mini | MCQA 과적합 방지와 일반 지시이행 보존 |

## 전처리 코드

새로 추가한 CPU-only 코드:

| file | 역할 |
|---|---|
| `scripts/build_cpt_repair_sft_jsonl.py` | legal bar와 KoTSQA를 answer-only/compact JSON/short rationale raw JSONL로 변환 |
| `scripts/validate_cpt_repair_sft_jsonl.py` | raw JSONL의 empty row, answer-only 비율, split 오염, 긴 응답 검증 |
| `scripts/validate_prepared_sft_arrays.py` | tokenized 배열의 vocab 범위, empty response labels, max length 검증 |
| `scripts/run_prepare_cpt_repair_sft_dryrun.sh` | 작은 dry-run 전처리와 검증 |
| `scripts/run_prepare_cpt_repair_sft_full.sh` | 200M-token full preprocess 준비 |
| `scripts/run_cpt_repair_sft_train_guarded.sh` | 학습 launcher. `ALLOW_TRAIN=1` 없이는 절대 실행하지 않음 |

## Dry-Run 검증 결과

실행 명령:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_prepare_cpt_repair_sft_dryrun.sh
```

실행 결과:

| 항목 | 값 |
|---|---:|
| raw rows | 700 |
| tokenized rows | 700 |
| total tokens | 329,767 |
| max sample len | 1,210 / 4,096 |
| token max | 124,900 |
| LFM vocab size | 125,017 |
| empty response rows | 0 |
| answer-only rows | 201 |
| answer-only ratio | 28.71% |
| warnings/errors | 0 |

raw category:

| category | rows |
|---|---:|
| `korean_mcqa_answer_only` | 200 |
| `korean_mcqa_json_compact` | 200 |
| `korean_mcqa_short_rationale` | 200 |
| `korean_evidence_qa_short_answer` | 100 |

검증 결론:

- LFM vocab 범위 문제 없음: `124900 < 125017`
- response-only label empty 없음
- 4k max sequence 안에 충분히 들어감
- answer-only 비율이 최소 기준 20%보다 높음
- KoTSQA는 train split만 사용

## Full Preprocess 결과

실행 명령:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
  bash scripts/run_prepare_cpt_repair_sft_full.sh
```

산출물:

```text
/home/work/.data/lfm2_ko_sft/prepared/repair_cpt/20260630_cpt_mcqa_repair_4k
```

요약:

| 항목 | 값 |
|---|---:|
| directory size | 1.1GB |
| raw repair rows | 6,750 |
| tokenized sample count | 188,493 |
| total LFM tokens | 131,607,379 |
| max seq length | 4,096 |
| avg sample len | 698.21 |
| max written len | 4,096 |
| token max | 124,907 |
| LFM vocab size | 125,017 |
| empty response rows | 0 |
| dedupe dropped | 3,234 |
| truncated rows | 1,528 |
| validation warnings/errors | 0 |

raw repair category:

| category | rows |
|---|---:|
| `korean_mcqa_answer_only` | 2,000 |
| `korean_mcqa_json_compact` | 2,000 |
| `korean_mcqa_short_rationale` | 2,000 |
| `korean_evidence_qa_short_answer` | 750 |

preservation exports:

| source | selected original tokens |
|---|---:|
| finance compact | 30,000,443 |
| Text2SQL DuckDB | 20,000,279 |
| SWE compact | 25,001,571 |
| reasoning compact | 20,000,189 |
| agent reasoning compact | 15,003,096 |
| behavior mini | 10,000,271 |

왜 200M이 아니라 131.6M인가:

- raw repair 데이터는 의도적으로 작고 compact하다.
- preservation source도 compact split 위주로 제한했다.
- 더 채우려면 full behavior, terminal/toolbench, 또는 장문 chat 데이터를 추가해야
  한다.
- 그러나 Stage2 실패 원인이 장문/도메인/chat SFT가 public MCQA scoring을 덮은
  것이라서, 1차 repair에서는 더 넣지 않는 게 맞다.

## 예상 완료 시간

GPU 학습 없이 CPU/network 전처리 기준이다. 시간은 I/O 부하에 따라 달라질 수 있다.

| 작업 | 예상 시간 |
|---|---:|
| dry-run raw + tokenized validation | 완료, 약 2초 |
| full raw repair JSONL 생성 | 완료 |
| preservation source export | 완료 |
| 131.6M-token LFM tokenization | 완료 |
| full validation/report 생성 | 완료 |
| 문서/모델카드 갱신 | 10-20분 |

나중에 명시적으로 학습 승인이 나면, 131.6M tokens / 4k / 8 H200 / effective
batch 128 기준으로 약 1.5-3시간 범위를 예상한다. 단, 이 문서 작성 시점에는
학습하지 않는다.

## 학습 Guard

학습 launcher는 기본적으로 실패한다.

```bash
bash scripts/run_cpt_repair_sft_train_guarded.sh
```

출력:

```text
Refusing to start training.
This launcher is intentionally guarded because the current instruction is:
do not use GPUs and do not train.
```

나중에 명시적으로 승인된 경우에만 다음처럼 실행한다.

```bash
ALLOW_TRAIN=1 bash scripts/run_cpt_repair_sft_train_guarded.sh
```

## 평가 Gate

학습 승인이 나더라도 full benchmark 전에 작은 gate로 중지 여부를 판단한다.

| gate | 기준 |
|---|---|
| IFEval | KO-CPT보다 크게 하락하면 중지 |
| GSM8K | KO-CPT 대비 회복/보존 확인 |
| BoolQ | KO-CPT 하락폭 확인 |
| ARC/PIQA | Base/CPT 대비 일반 추론 붕괴 확인 |
| Global MMLU KO | 한국어 지식 보존 확인 |
| KMMLU direct hard | 다지선다 repair 핵심 gate |
| MMLU-ProX Lite KO | 한국어 hard exact-answer gate |

## 왜 이게 맞는가

Stage2 KO-SFT 실패 결과는 지식 부족 신호가 아니라 형식/확률분포 붕괴 신호다.
KO-CPT는 이미 공개 벤치에서 더 강하다. 그러므로 새 raw corpus CPT를 더 넣는 것은
문제와 직접 맞지 않고 시간이 오래 걸린다. 반면 작은 repair SFT는 원인에 직접
맞는다.

핵심은 “더 많이 학습”이 아니라 “CPT의 좋은 분포를 덜 건드리면서, 정답 형식과
다지선다 선택 행동만 고치기”다.
