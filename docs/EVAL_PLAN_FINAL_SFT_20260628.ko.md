# LFM2.5 KO SFT 최종 평가 계획

작성 시각: 2026-06-28

## 목표

학습이 모두 끝난 뒤 `base -> CPT -> SFT`를 같은 조건으로 비교한다.

- Base: `LiquidAI/LFM2.5-8B-A1B`
- CPT: `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`
- SFT: `LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT`

## Liquid 공식 카드 기준

LiquidAI 모델 카드가 공개한 LFM2.5-8B-A1B 주요 benchmark는 다음이다.

| category | official benchmark | local status |
|---|---|---|
| knowledge / hallucination | AA-Omniscience | 별도 harness 필요 |
| instruction following | IFEval | lm-eval 가능: `ifeval` |
| instruction following | IFBench | 별도 harness 필요 |
| instruction following | Multi-IF | 별도 harness 필요 |
| math | MATH500 | lm-eval 가능: `minerva_math500` |
| math | AIME25 | lm-eval 가능: `aime25` |
| function calling | BFCLv3 | 별도 harness 필요 |
| function calling | BFCLv4 | 별도 harness 필요 |
| agentic tasks | Tau2 Telecom | 별도 harness 필요 |
| agentic tasks | Tau2 Retail | 별도 harness 필요 |

공식 LFM 모델 카드:

- <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>
- Liquid blog: <https://www.liquid.ai/blog/lfm2-5-8b-a1b>

## 바로 돌릴 lm-eval/vLLM 항목

로컬 `lm-eval` registry에서 확인된 항목만 우선 자동 queue에 넣는다.

| area | task group |
|---|---|
| instruction following | `ifeval` |
| math | `minerva_math500` |
| math | `aime25` |
| grade-school reasoning | `gsm8k` |
| general MCQA | `arc_challenge,hellaswag,truthfulqa_mc2` |
| Korean MMLU | `global_mmlu_ko` |
| Korean MMLU full | `global_mmlu_full_ko` |
| Korean hard MCQA | `kmmlu_direct_hard` |
| Korean hard CoT MCQA | `kmmlu_cot_hard` |
| Korean benchmark suite | `kobest` |
| Korean language knowledge | `haerae` |
| Korean reading | `belebele_kor_Hang` |

설정 파일:

- `configs/eval_models_final_sft_20260628.txt`
- `configs/eval_task_groups_final_sft_lm_eval_20260628.txt`

## 별도 harness가 필요한 항목

다음은 local `lm-eval`에 task가 없어서 별도 설치/구현이 필요하다.

- AA-Omniscience
- IFBench
- Multi-IF
- BFCLv3/BFCLv4
- Tau2 Telecom/Retail

이 항목들은 최종 SFT 모델이 나온 뒤 GPU를 비우고 별도 환경을 만든다. 우선순위는:

1. BFCLv4: tool/function calling 보존 확인
2. Tau2 Telecom/Retail: agentic multi-turn tool-use 확인
3. IFBench/Multi-IF: 복잡 지시 추종 확인
4. AA-Omniscience: hallucination/abstention 확인

## 자동 실행

Stage2 학습 완료 후 바로 lm-eval/vLLM 평가를 시작한다.

```bash
tmux attach -t lfm2ko_chain_final_eval_after_stage2_20260628
```

실행되는 명령:

```bash
RUN_ID=20260630_final_sft_lm_eval \
MODELS_FILE=configs/eval_models_final_sft_20260628.txt \
TASK_GROUPS_FILE=configs/eval_task_groups_final_sft_lm_eval_20260628.txt \
GPU_COUNT=8 \
MAX_MODEL_LEN=8192 \
bash scripts/run_vllm_eval_8gpu_queue.sh
```

## 결과 반영

결과 파일:

```text
/home/work/.data/lfm2_ko_sft/eval/20260630_final_sft_lm_eval/SUMMARY.md
```

완료 후 해야 할 일:

1. `SUMMARY.md`에서 base/CPT/SFT를 같은 task 기준으로 표로 정리한다.
2. 한국어 성능이 오른 항목과 떨어진 항목을 분리한다.
3. 모델 카드 상단에 주요 점수표를 추가한다.
4. README와 이 문서에 최종 결과와 추가 후속 학습 후보를 반영한다.
