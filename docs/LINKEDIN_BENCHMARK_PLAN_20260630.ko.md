# LinkedIn Public Benchmark Plan - 2026-06-30

목표는 홍보글에 쓸 수 있는 숫자를 빠르게 확보하는 것이다. 따라서 이미 CPT
모델 카드에서 Base/CPT 비교가 끝난 항목은 다시 돌리지 않는다. Stage2 KO-SFT만
같은 task로 full 평가하고, `configs/benchmark_reference_cpt_card_20260630.json`의
Base/CPT 기준값과 합쳐 비교표를 만든다.

## 바로 돌릴 Stage2 SFT-only full 평가

실행:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_stage2_linkedin_full_eval.sh
```

모델:

- `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage2-4k-diverse-kotsqa-20260628/final_full`

결과:

```text
/home/work/.data/lfm2_ko_sft/eval/20260630_stage2_linkedin_full/LINKEDIN_BENCHMARK_SUMMARY.md
```

평가 항목:

| group | tasks | purpose |
|---|---|---|
| instruction | `ifeval`, `leaderboard_ifeval` | CPT에서 오른 지시 준수 보존 |
| reasoning/general | `gsm8k`, `boolq`, `arc_challenge`, `piqa` | CPT의 reasoning/commonsense gain 보존 |
| Korean gain subjects | `global_mmlu_full_ko_medical_genetics`, `nutrition`, `philosophy`, `miscellaneous` | CPT에서 오른 한국어 과목 보존 |
| Korean regression subjects | `professional_medicine`, `high_school_statistics`, `astronomy`, `high_school_computer_science`, `jurisprudence` | CPT에서 내려간 과목 회복 |
| Korean MCQA/extraction | `kmmlu_direct_hard`, `kmmlu_direct_hard_stem`, `mmlu_prox_lite_ko` | 다지선다/정답 라벨 회복 |
| professional MCQA | `mmlu_prox_lite_ko` | 한국어 전문 다지선다/정답 추출 회복 |

이 평가가 끝나면 summary script가 Base/CPT reference와 KO-SFT 결과를 병합해서
`SFT-Base`, `SFT-CPT` delta를 계산한다.

## Agentic 이후로 미루는 supplement

LiquidAI 원본 `LFM2.5-8B-A1B` 모델 카드는 IFEval, IFBench, Multi-IF, MATH500,
AIME25, BFCLv3/v4, Tau2, AA-Omniscience를 공식 축으로 보고한다. 이 중 현재
로컬 lm-eval에서 즉시 재현 가능한 느린 supplement는 다음만 자동 체인 뒤쪽에 둔다.

```bash
bash scripts/run_official_supplement_after_agentic_eval.sh
```

평가 항목:

- `minerva_math500`
- `aime25`
- `truthfulqa_mc2`
- `mmlu_pro_law`
- `mmlu_pro_economics`

BFCLv3/v4, Tau2, IFBench, Multi-IF, AA-Omniscience는 별도 external harness가
필요하므로 이번 빠른 Stage2 공개 비교 경로에서는 제외한다.

## 체인 순서

1. Stage2 학습 완료.
2. Stage2 `final_full` 안정화.
3. Stage2 Hub upload는 background로 시작.
4. GPU는 즉시 Stage2 LinkedIn SFT-only full eval로 이동.
5. Agentic/Fable SFT.
6. Agentic eval + harness smoke.
7. official supplement eval.
8. 모델 카드, README, LinkedIn 표 반영.
