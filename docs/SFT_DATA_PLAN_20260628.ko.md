# LFM2.5-8B-A1B-KO-SFT 데이터 계획

작성 시각: 2026-06-28

## 목표

`LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT`는 CPT 모델 위에 SFT를 얹어 다음 성능을 동시에 노린다.

- 한국어 법률/금융 질의응답 강화
- 변시/법률/회계/금융 다지선다의 정답 형식 복구
- LFM2.5 chat template, tool-use, terminal JSON 동작 보존
- 코딩/SWE와 영어 reasoning 성능 보존

## 현재 진행 상태

2026-06-30 사후 업데이트: 아래 데이터 체인은 학습과 업로드까지 끝났지만, 공개
벤치 결과는 실패로 판정한다. Stage2 KO-SFT는 대부분의 public benchmark에서
KO-CPT보다 낮았고, Stage3 Agentic/Fable도 benchmark repair에는 실패했다. 상세
원인은 `docs/SFT_AGENTIC_FAILURE_ANALYSIS_20260630.ko.md`에 정리했다.

| 항목 | 상태 | 경로 / 수치 |
|---|---|---|
| Stage0 legal | 완료 | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0-legal-20260628/final_full` |
| Stage0b finance/Text2SQL | 완료/업로드 | 280,000 samples, 58,090,087 tokens, 2,188 planned steps |
| Stage1 4k finance/Text2SQL | 완료/업로드 | 2,302,304 samples, 1,285,864,494 tokens, 17,987 planned steps |
| Stage1 8k legal/terminal | 완료/업로드 | 1,600,835 samples, 1,658,848,754 tokens |
| Stage2 diverse KO/SWE/reasoning | 완료 | 1,467,864 samples, 1,364,349,642 tokens, raw CPT corpus 제외 |
| Stage2 plus KoTSQA | 완료/업로드 | 1,468,598 samples, 1,364,863,776 tokens; KoTSQA train split만 추가 |
| SFT 합계 | 완료 | 1.286B + 1.659B + 1.364864B = 4.309577B tokens |

Stage1 4k의 2026-06-28 16:41 KST 기준 실측 속도는 약 57-59 examples/sec,
약 2.2초/update step이었다. 이 수치는 사후 감사용 기록이다.

## LFM 포맷 기준

LFM2.5 SFT는 Liquid 문서의 대화 형식을 맞춰야 한다.

- Prompting guide:
  <https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting>
- Chat template:
  <https://docs.liquid.ai/lfm/key-concepts/chat-template>
- Tool use:
  <https://docs.liquid.ai/lfm/key-concepts/tool-use>
- Base model:
  <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>

중요한 구현 기준:

- `system`, `user`, `assistant`, `tool` role을 유지한다.
- 툴콜 데이터는 Liquid 문서의 `<|tool_call_start|>...<|tool_call_end|>` 동작을
  보존하는 방향으로 변환한다.
- assistant 응답만 loss를 주는 response-only labels를 사용한다.
- 기존 prepared 데이터 일부는 LFM vocab 범위를 넘는 token id를 포함해서
  CUDA device-side assert를 냈다. 그래서 `scripts/prepare_lfm_chat_sft_data.py`
  로 LFM tokenizer 기준 데이터를 다시 만들었다.

## 확정 로컬 Prepared 데이터

| stage | source | path | source/origin | size | tokens | ratio before dedupe |
|---|---|---|---|---:|---:|---:|
| 8k | Terminal ToolBench Full | `/home/work/.data/hrm_text_prepared/kohrm_sft_lfm25_terminal_toolbench_full_v1` | `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` 계열 terminal/tool conversations | 5.7G | 1.509B | 42.2% |
| 4k | Behavior Core | `/home/work/.data/hrm_text_prepared/kohrm_sft_behavior_core_v1` | terminal/tool/SWE/reasoning/legal/finance balanced core | 1.2G | 0.285B | 8.0% |
| 4k | SWE Zero | `/home/work/.data/hrm_text_prepared/sft_swe_zero_v1` | SWE/coding repair data | 720M | 0.183B | 5.1% |
| 4k | Korean Legal Tasks | `/home/work/.data/hrm_text_prepared/korean_legal_tasks_full_v1` | Korean legal task corpus | 2.5G | 0.629B | 17.6% |
| 4k | Korean Finance | `/home/work/.data/hrm_text_prepared/sft_bcai_finance_kor_v1` | BCAI Korean finance instruction corpus | 3.3G | 0.858B | 24.0% |
| 4k | Text2SQL Clean DuckDB | `/home/work/.data/hrm_text_prepared/kohrm_sft_text2sql_core_clean_duckdb_v1` | structured Text2SQL/code-like reasoning | 495M | 0.115B | 3.2% |

Prepared subtotal: about **3.579B tokens** before dedupe/weighting.

## LFM Chat Prepared 결과

| split | local path | max seq | samples | tokens | status |
|---|---|---:|---:|---:|---|
| Stage0 legal | `/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage0_legal` | 8192 | 8,747 | 35,068,923 | trained |
| Stage0b fast finance/Text2SQL | `/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage0b_fast_mix_4k_finance_text2sql` | 4096 | 280,000 | 58,090,087 | trained/uploaded |
| Stage1 4k finance/Text2SQL | `/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage1_ko_finance_terminal_text2sql_4k_finance_text2sql` | 4096 | 2,302,304 | 1,285,864,494 | ready |
| Stage1 8k legal/terminal | `/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage1_ko_finance_terminal_text2sql_8k_legal_terminal` | 8192 | 1,600,835 | 1,658,848,754 | ready |
| Stage2 diverse KO/SWE/reasoning | `/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage2_diverse_ko_swe_reasoning_4k` | 4096 | 1,467,864 | 1,364,349,642 | ready |
| Stage2 plus KoTSQA | `/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage2_plus_kotsqa_4k` | 4096 | 1,468,598 | 1,364,863,776 | ready |

## JSONL SFT Shards

| source | path | source/origin | size | rows | purpose |
|---|---|---|---:|---:|---|
| Legal Source Agent | `/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/006_ko_legal_source_agent_sft_20260621.jsonl` | Korean legal source-grounded SFT | 119M | 5,999 | legal citation/source behavior |
| Legal RAG Round15 | `/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/007_ko_legal_rag_agent_sft_round15_v2.jsonl` | Korean legal RAG agent SFT | 15M | 749 | legal RAG behavior |
| Current Law Bar JSON | `/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/008_current_law_bar_json_answer_sft_20260621.jsonl` | legalize-kr/law.go.kr based bar-style JSON answers | 8.6M | 2,000 | MCQA exact-answer format |
| Terminal ToolBench JSONL QA | `/home/work/.data/lfm2_ko_cpt/datasets/shards_full_lfmstyle_20260627/009_lfm25_terminal_toolbench_hrm_turns_v1.jsonl` | terminal/tool conversation source JSONL | 4.6G | 326,785 | QA/dedupe reference; primary training uses prepared 8k version |

## 외부/추가 후보

| source | URL | usage decision |
|---|---|---|
| legalize-kr | <https://github.com/legalize-kr> | 법률 출처 및 법령/판례/시험형 데이터 출처 명시 |
| LLM-Ko-Datasets | <https://github.com/gyunggyung/LLM-Ko-Datasets> | 한국어 QA, SFT, domain 후보 인덱스 |
| KoTSQA v2 | <https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0> | `train` split만 Stage2 보강 SFT에 사용, `test`는 평가용 보류 |

추가 데이터는 평가셋 오염 여부를 먼저 확인해야 한다. 특히 KMMLU, Global MMLU
Korean, 법률 MCQA, 금융 MCQA에 직접 들어갈 항목은 train mix에서 제외한다.

## Preprocessing Policy

- Do not drop a domain just because it overlaps. Remove exact duplicate examples by hash, but preserve each category.
- Keep 8k and 4k prepared sets separate because `max_seq_len` differs (`8193` vs `4097`).
- Stage0: 작은 legal warmup으로 포맷과 저장을 검증한다.
- Stage0b: finance/Text2SQL fast mix로 8 GPU direct DDP를 검증한다.
- Stage1: 4k finance/Text2SQL과 8k legal/terminal을 분리해서 풀 SFT한다.
- Stage2: 평가 결과를 보고 legal/finance/MCQA/terminal/tool 비율을 재조정한다.
- JSONL legal/bar shards are converted into `instruction/response/condition` and tokenized with the LFM tokenizer.
- Final training mix will use source weights, not one giant blind concat, so we can adjust after base/CPT/SFT eval.

## 2026-06-30 결과 기반 데이터 해석

이번 데이터 준비는 LFM tokenizer, response-only labels, chat template, tool-call
role 보존 측면에서는 맞았다. 문제는 포맷이 아니라 데이터 목적과 평가 목적의
불일치였다.

| 데이터군 | 의도 | 실제 공개 벤치 관점의 문제 |
|---|---|---|
| finance/accounting/Text2SQL | 도메인 설명과 구조화 출력 | BoolQ 외 public MCQA repair에는 직접적이지 않음 |
| legal/terminal 8k | 장문 법률 근거, 터미널/툴 trace | 장문 답변과 절차형 행동을 강화하지만 option-only scoring을 약화시킬 수 있음 |
| diverse KO/SWE/reasoning | 도메인 편향 완화와 코딩/추론 보강 | answer-only/MCQA 비중이 부족해 KMMLU/MMLU-ProX 하락을 막지 못함 |
| KoTSQA train | 근거 QA와 false-premise correction | 좋은 보강 데이터지만 다지선다 repair 데이터는 아님 |
| Agentic/Fable | 로그/문서/터미널 agent behavior | public benchmark repair 목적과 다르고 규모도 7.12M tokens로 작음 |

따라서 다음 데이터 계획은 대용량 SFT 재반복이 아니라, KO-CPT 위에서 작은
정확답/다지선다 repair set을 별도로 준비하는 방향이어야 한다.

필요한 repair 데이터 형태:

- 한국어 question + choices + answer-only 출력
- 짧은 rationale 뒤 `Final answer: A` 같은 분리된 최종 답
- 법률/금융/회계 MCQA의 train split 또는 합성 문제
- KoTSQA train 기반의 근거 QA는 유지하되, 최종 답 형식을 짧게 고정한 변형
- JSON exact-answer와 Text2SQL exact-output은 과다 장문 설명 없이 별도 비율 제한

넣지 말아야 할 것:

- 평가 test split
- 이미 CPT에서 충분히 사용한 raw corpus
- 장문 chat-only 데이터의 무제한 추가
- Agentic/Fable trace를 main public benchmark repair에 섞는 것

## 학습 명령

현재 동작하는 경로는 direct `torchrun` DDP이다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
DATASET_PATH=/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage0b_fast_mix_4k_finance_text2sql \
MODEL_PATH=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0-legal-20260628/final_full \
RUN_ID=stage0b-finance-text2sql-20260628 \
STAGE_NAME=stage0b_finance_text2sql \
MAX_SEQ_LENGTH=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=2 \
PUSH_TO_HUB=1 \
bash scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh
```

Batch 계산:

- GPU 수: 8
- per-device batch: 2
- gradient accumulation: 8
- effective batch: 128 sequences/update

Stage0b는 280,000 samples라서 `ceil(280000 / 128) = 2188` update steps이다.

Stage1 전체를 같은 effective batch 128로 1 epoch 돌리면:

- 4k split: `ceil(2302304 / 128) = 17,987` steps
- 8k split: `ceil(1600835 / 128) = 12,507` steps
- 단순 합계: 약 30,494 update steps

2026-06-28 16:41 KST 기준 자동 진행 순서와 예상 완료 시간:

| order | stage | tokens | estimated completion |
|---:|---|---:|---|
| 1 | Stage1 4k finance/Text2SQL | 1.286B | 2026-06-29 03:15-03:45 KST |
| 2 | Stage1 8k legal/terminal | 1.659B | 2026-06-29 19:30-2026-06-30 01:30 KST |
| 3 | Stage2 4k diverse plus KoTSQA | 1.364864B | 2026-06-30 07:30-14:00 KST |

8k split은 토큰 길이가 길어서 4k split보다 느릴 수 있다. 완료 시간은 각 stage가
시작되면 `train_log.jsonl`의 실제 속도로 갱신한다.

자동 체인:

```bash
tmux attach -t lfm2ko_chain_after_stage1_20260628
tmux attach -t lfm2ko_chain_stage2_after_8k_20260628
tmux attach -t lfm2ko_prep_kotsqa_stage2_plus_20260628
```

## Stage2 Diverse SFT 계획

사용한다:

| category | source | old-token estimate |
|---|---|---:|
| Korean domain | `kohrm_sft_korean_domain_core_v1` | 0.100B |
| behavior mix | `kohrm_sft_behavior_core_v1` | 0.285B |
| coding/SWE | `sft_swe_zero_v1` | 0.183B |
| coding/SWE | `sft_swe_glm_mix_v1` | 0.251B |
| coding/SWE compact | `kohrm_sft_comp_swe_zero_30m_v1` | 0.030B |
| reasoning | `sft_glm_reasoning_v1` | 0.068B |
| reasoning/agent | `hf_extra_reasoning_agent_mm_v1` | 0.113B |
| reasoning/agent compact | `kohrm_sft_comp_agent_reasoning_25m_v1` | 0.025B |
| finance compact | `kohrm_sft_comp_finance_50m_v1` | 0.050B |
| legal compact | `kohrm_sft_comp_korean_legal_50m_v1` | 0.050B |
| Text2SQL | `kohrm_sft_text2sql_core_clean_duckdb_v1` | 0.115B |

LFM tokenizer 변환 완료본은 1,364,349,642 tokens이다.

KoTSQA 보강:

- Source: <https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0>
- 학습 사용: `train` split only
- 평가 보류: `test` split
- 변환 결과: 750 samples, 514,134 tokens
- merge 결과: Stage2 plus KoTSQA 1,468,598 samples, 1,364,863,776 tokens
- 변환 스크립트: `scripts/convert_kotsqa_to_lfm_sft_jsonl.py`
- 준비/merge 스크립트: `scripts/run_prepare_lfmchat_kotsqa_stage2_plus.sh`
- 목적: 한국어 표/문서 근거 QA, multi-hop QA, false-premise correction 보강

제외한다:

- `kowiki_raw_full_v1`
- `korean_legal_raw_full_v1`
- `korean_admrule_precedent_raw_full_v1`
- `koterm_pretrain_mix_v1`
- `koterm_hrm_cleaned_full_nocap_v1`
- `koterm_hrm_cleaned_fastcap_stage1_v1`

이들은 SFT가 아니라 CPT/mid-training 성격이 강하다. 이미 CPT에서 한국어 지식
주입을 했으므로 이번 SFT에는 넣지 않는다.

전처리 명령:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
CONCURRENCY=4 bash scripts/run_prepare_lfmchat_stage2_diverse.sh
```

현재 tmux:

```bash
tmux attach -t lfm2ko_sft_stage2_prep_20260628
```

## Evaluation Tie-In

The same vLLM/lm-eval matrix should be run for:

- `LiquidAI/LFM2.5-8B-A1B`
- `LFM2.5-8B-A1B-KO-CPT-FULL`
- `LFM2.5-8B-A1B-KO-SFT`

Primary score table:

- Liquid official lineage: AA-Omniscience, IFEval, IFBench, Multi-IF, MATH500,
  AIME25, BFCLv3, BFCLv4, Tau2 Telecom, Tau2 Retail
- Korean: Global MMLU KO, KMMLU hard
- Domain: Korean legal/accounting/finance targeted subsets
- Format: bar-style JSON answer extraction, Text2SQL exact match, tool-call syntax

SFT success condition: improve Korean domain and MCQA extraction while not destroying CPT gains on instruction following and GSM8K.

평가 게이트:

1. Stage0b final 직후: `ifeval`, `gsm8k`, `truthfulqa_mc2`, 한국어 샘플 generation
   smoke test만 빠르게 본다.
2. Stage1 4k 직후: 한국어 금융/법률 short probes, Text2SQL exact-format probe,
   `ifeval` subset을 본다.
3. Stage1 8k 직후: tool-call syntax, terminal/tool behavior, 법률 장문 source behavior를
   본다.
4. Stage2 직후: SWE/coding, reasoning, 한국어 일반 질의, MCQA 형식이 무너지지 않았는지
   full 비교한다.

실행 스크립트:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

# 단일 모델/단일 GPU
CUDA_VISIBLE_DEVICES=0 \
MODEL_ID=LiquidAI/LFM2.5-8B-A1B \
TASKS=ifeval,gsm8k \
bash scripts/run_vllm_lm_eval_matrix.sh

# 8 GPU 병렬 queue. 현재는 사용자 지시에 따라 실행하지 않는다.
bash scripts/run_vllm_eval_8gpu_queue.sh
```

기본 설정 파일:

- `configs/eval_models_20260628.txt`
- `configs/eval_task_groups_20260628.txt`

## 다음 Post-Training 제안

1. 지금은 사용자 지시에 따라 추가 학습을 하지 않는다.
2. 실패한 Stage2/Stage3 checkpoint에서 이어가지 않는다.
3. 재개한다면 KO-CPT에서 작은 MCQA/정확답 repair SFT를 새로 시작한다.
4. repair set은 answer-only, final option, 짧은 rationale, JSON exact-answer를
   중심으로 만들고 장문 chat-only 비중을 제한한다.
5. gate에서 KO-CPT보다 떨어지면 즉시 중지한다.
