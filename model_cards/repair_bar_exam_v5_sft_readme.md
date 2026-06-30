---
base_model:
- LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT
- LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
language:
- ko
- en
tags:
- lfm
- korean
- sft
- legal
- bar-exam
- diagnostic
pipeline_tag: text-generation
license: other
---

# LFM2.5-8B-A1B-KO-CPT-Repair-BarExamV5-SFT

This is a diagnostic legal-context SFT checkpoint trained on top of
[`LFM2.5-8B-A1B-KO-CPT-Repair-SFT`](https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT).
It is published to preserve the experiment and should not be presented as the
main public benchmark model.

The current recommendation is still the KO-CPT checkpoint:
[`LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`](https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL).

- SFT GitHub: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT GitHub: <https://github.com/gyunggyung/LFM25-KO-CPT>
- Repair SFT: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-Repair-SFT>

## Training Data

This stage was designed to teach a narrow legal-context behavior:
given a bar-exam style question plus a curated evidence packet, read the legal
grounds, compare choices, and output a normalized answer.

| item | value |
|---|---:|
| prepared dataset | `/home/work/.data/lfm2_ko_sft/prepared/bar_exam_v5/20260630_bar_exam_v5_context_solver_8192/lfm_chat_8192` |
| samples | 6,374 |
| LFM tokens | 5,863,863 |
| max sequence length | 8,192 |
| epochs | 1 |
| learning rate | 5e-7 |
| planned optimizer steps | 57 |

The data includes 1-14th bar-exam safe numeric/symbol answer variants, current
law-bar synthetic tasks, 15th v5 evidence-reading procedure samples, legal
search-first-action samples, full v5 grounded solutions, and compact v5 grounded
answers.

## Gate Evaluation

Evaluation root:

```text
/home/work/.data/lfm2_ko_sft/eval/bar_exam_v5_sft_gate_20260630T1306KST
```

Base/CPT reference values are copied from the KO-CPT model card where available.
Some metrics differ from this gate run, so rows marked with a metric mismatch
should be read directionally rather than as strict apples-to-apples comparisons.

| task | Base | KO-CPT | Repair-SFT | BarExamV5-SFT | v5 metric | verdict |
|---|---:|---:|---:|---:|---|---|
| BoolQ | 0.6544 | 0.7902 | 0.663303 | 0.658716 | `acc,none` | below CPT, slightly below Repair |
| ARC-Challenge | 0.3771 | 0.4241 | 0.211604 | 0.209898 | `acc,none`; CPT uses `acc_norm` | below Base/CPT |
| IFEval | 0.2921 | 0.3216 | 0.181146 | 0.188540 | strict prompt acc; CPT uses loose prompt acc | small gain vs Repair, below CPT |
| Global MMLU KO jurisprudence | 0.2870 | 0.2685 | 0.250000 | 0.296296 | `acc,none` | narrow gain vs Repair/CPT |
| KMMLU direct hard | 0.2015 | 0.1720 | 0.102339 | 0.101608 | `exact_match,none`; CPT card uses `acc,none` | below Base/CPT |
| KMMLU direct hard law | n/a | n/a | 0.170000 | 0.190000 | `exact_match,none` | small legal-category gain |
| MMLU-ProX Lite KO | 0.2585 | 0.1667 | 0.091837 | 0.068027 | `exact_match,custom-extract` | worsened vs Repair |
| MMLU-ProX Lite KO law | n/a | n/a | 0.104167 | 0.020833 | `exact_match,custom-extract` | severe extraction regression |

## Interpretation

This stage produced a narrow jurisprudence gain, but it harmed broader
multiple-choice/exact-extraction behavior. That is consistent with the training
objective: long legal solution traces can push the model toward verbose legal
reasoning and away from concise answer extraction.

Final lesson: Korean bar-exam style solving should be treated as an
evidence-grounded workflow problem, not as standalone SFT memorization. The
model needs curated legal context, explicit O/X judgment, symbol-to-number
choice mapping, and strict extraction gates. This diagnostic SFT showed that
full-solution traces alone are not enough for reliable open-model bar-exam
performance.

For the next attempt, this should be split into either a separate legal-context
adapter/model with its own held-out v5 evaluator, or a much smaller late-stage
injection after a proven repair checkpoint. It should not be merged into the
main public benchmark line without an early gate that preserves KMMLU,
MMLU-ProX, GSM8K, ARC, BoolQ, and IFEval.
