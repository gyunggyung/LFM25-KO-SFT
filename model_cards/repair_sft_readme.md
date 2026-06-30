---
base_model:
- LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
language:
- ko
- en
tags:
- lfm
- korean
- sft
- diagnostic
- legal
pipeline_tag: text-generation
license: other
---

# LFM2.5-8B-A1B-KO-CPT-Repair-SFT

This is a diagnostic repair-SFT checkpoint trained from
[`LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`](https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL).
It is published for reproducibility, not as the recommended benchmark model.

The current recommendation is still the KO-CPT checkpoint. This repair SFT did
not recover the CPT public benchmark profile.

- SFT GitHub: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT GitHub: <https://github.com/gyunggyung/LFM25-KO-CPT>
- CPT model: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL>

## Training Data

The run used a small answer-format repair mixture instead of the previous large
general SFT line.

| item | value |
|---|---:|
| prepared dataset | `/home/work/.data/lfm2_ko_sft/prepared/repair_cpt/20260630_cpt_mcqa_repair_4k/lfm_chat_4k` |
| samples | 188,493 |
| LFM tokens | 131,607,379 |
| max sequence length | 4,096 |
| epochs | 1 |
| learning rate | 1e-6 |
| global effective batch | 128 |
| planned optimizer steps | 1,473 |

The mixture focused on Korean MCQA answer formatting, KoTSQA train evidence QA,
finance/Text2SQL preservation, SWE/coding preservation, and compact reasoning
preservation. It intentionally avoided continuing from the failed Stage2/Stage3
SFT checkpoints.

## Gate Evaluation

Evaluation root:

```text
/home/work/.data/lfm2_ko_sft/eval/repair_sft_gate_20260630T1306KST
```

Base/CPT reference values are copied from the KO-CPT model card where available.
Some metrics differ from this gate run, so rows marked with a metric mismatch
should be read directionally rather than as strict apples-to-apples comparisons.

| task | Base | KO-CPT | Repair-SFT | repair metric | verdict |
|---|---:|---:|---:|---|---|
| BoolQ | 0.6544 | 0.7902 | 0.663303 | `acc,none` | above Base, far below CPT |
| ARC-Challenge | 0.3771 | 0.4241 | 0.211604 | `acc,none`; CPT uses `acc_norm` | below Base/CPT |
| GSM8K | 0.4845 | 0.5701 | 0.329795 | `strict-match`; CPT uses `flexible-extract` | below Base/CPT |
| IFEval | 0.2921 | 0.3216 | 0.181146 | strict prompt acc; CPT uses loose prompt acc | below Base/CPT |
| Global MMLU KO jurisprudence | 0.2870 | 0.2685 | 0.250000 | `acc,none` | below Base/CPT |
| KMMLU direct hard | 0.2015 | 0.1720 | 0.102339 | `exact_match,none`; CPT card uses `acc,none` | below Base/CPT |
| MMLU-ProX Lite KO | 0.2585 | 0.1667 | 0.091837 | `exact_match,custom-extract` | below Base/CPT |

## Interpretation

The repair attempt did not solve the regression seen in the earlier KO-SFT
line. The likely failure mode is still answer distribution drift: response-only
chat SFT changed the model toward verbose assistant behavior and away from
short exact-answer extraction required by public MCQA tasks.

For public benchmark reporting, prefer the KO-CPT checkpoint. If another repair
attempt is made, it should be much smaller, use a lower learning rate, and stop
early based on 100/300/500-step gate evaluations.
