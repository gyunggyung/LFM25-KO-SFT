---
base_model:
- LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT
- LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
license: other
language:
- ko
- en
tags:
- lfm
- korean
- agentic
- terminal
- fable
- tool-use
- diagnostic
pipeline_tag: text-generation
---

# LFM2.5-8B-A1B-KO-Agentic-SFT

Agentic/Fable diagnostic SFT follow-up for
`LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT`.

- SFT model repo: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT>
- CPT model repo: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL>
- GitHub SFT repo: <https://github.com/gyunggyung/LFM25-KO-SFT>
- GitHub CPT repo: <https://github.com/gyunggyung/LFM25-KO-CPT>
- Base model: <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>
- Liquid chat template docs: <https://docs.liquid.ai/lfm/key-concepts/chat-template>
- Liquid tool-use docs: <https://docs.liquid.ai/lfm/key-concepts/tool-use>

## Status

This model is uploaded for reproducibility and diagnostic analysis. It should
not be treated as a public benchmark improvement over KO-CPT. It adds a small
Agentic/Fable behavior stage after the failed Stage2 KO-SFT line.

| item | value |
|---|---:|
| samples | 3,943 |
| tokens | 7,124,298 |
| max sequence length | 8192 |
| training method | full-parameter response-only SFT |
| source checkpoint | Stage2 KO-SFT final |
| purpose | terminal/log/document grounded behavior probe |

## Verdict

The Stage3 Agentic/Fable run is not a successful public benchmark repair. It
shows small partial recovery on a few diagnostic slices, but it does not restore
the broad benchmark quality that KO-CPT had before SFT.

| task | Stage2 KO-SFT | Agentic/Fable | change |
|---|---:|---:|---:|
| Global MMLU KO limit50 | 0.244681 | 0.251773 | +0.007092 |
| Global MMLU KO medical limit50 | 0.361111 | 0.416667 | +0.055556 |
| IFEval strict limit50 | 0.1000 | 0.1000 | +0.0000 |
| KMMLU direct hard limit50 | 0.113407 | 0.109734 | -0.003673 |
| MMLU-Pro law | 0.134423 | 0.150772 | +0.016349 |
| MMLU-Pro economics | 0.323460 | 0.331754 | +0.008294 |
| TruthfulQA MC2 | 0.474975 | 0.476824 | +0.001849 |
| BoolQ | 0.6664 | 0.664220 | -0.002180 |
| GSM8K exact | 0.3381 | 0.360879 | +0.022779 |

Interpretation:

- The model slightly recovers some law/economics/global-mmlu diagnostic slices.
- It does not improve IFEval.
- KMMLU direct hard remains weak.
- GSM8K recovers only from the failed Stage2 level and remains below KO-CPT and
  the original base reference.

## Data

Agentic/Fable data sources:

| source | local source | purpose |
|---|---|---|
| Fable5 Korean traces | `fable_distillation/datasets_ko/fable5_ko_sft_20260624.jsonl` | terminal, search, file reading, error-fix traces |
| Helio Korean traces | `fable_distillation/datasets_ko/helio_ko_sft_20260628.jsonl` | long-form reasoning traces |
| local grounded examples | generated from this workspace | README/runbook/train-log/git/vLLM diagnosis examples |

Public dataset releases:

- Raw Agentic/Fable data:
  <https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-Raw>
- Tokenized Agentic/Fable 8k arrays:
  <https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-8K>

## Why It Did Not Fix Benchmarks

This stage is only 7.12M tokens and is focused on behavior traces: reading logs,
following repository docs, planning terminal commands, and explaining evidence.
Those examples are not direct repair data for Korean multiple-choice likelihood,
exact-answer extraction, or option-only output.

The previous Stage2 KO-SFT checkpoint had already moved away from the KO-CPT
benchmark distribution. Training a small agentic trace set on top of that did
not restore the lost answer-token scoring behavior.

## Recommended Use

Use this checkpoint only for inspecting the Agentic/Fable behavior experiment.
For public Korean benchmark quality, prefer:

<https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL>

If future work resumes, start from KO-CPT and run a small MCQA/answer-format
repair SFT with strict gates instead of continuing this checkpoint.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a careful terminal and log analysis assistant."},
    {"role": "user", "content": "다음 학습 로그에서 loss가 갑자기 튀는 원인을 어떻게 확인할지 단계별로 말해줘."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Korean Summary

이 모델은 Stage2 KO-SFT 위에 Fable/문서/로그 기반 agentic SFT를 소량 얹은
진단용 모델입니다. 공개 벤치 개선 모델이 아닙니다. 일부 항목은 Stage2보다
조금 회복했지만, KO-CPT가 가진 공개 벤치 성능을 되찾지 못했습니다.
