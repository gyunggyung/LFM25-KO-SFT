---
base_model:
- LiquidAI/LFM2.5-8B-A1B
- LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
license: other
language:
- ko
- en
tags:
- lfm
- korean
- legal
- finance
- tool-use
- terminal
- sft
pipeline_tag: text-generation
---

# LFM2.5-8B-A1B-KO-SFT

Korean full-parameter SFT continuation of
`LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`, based on
`LiquidAI/LFM2.5-8B-A1B`.

- GitHub: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT GitHub: <https://github.com/gyunggyung/LFM25-KO-CPT>
- CPT base checkpoint: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL>
- Agentic follow-up repo: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT>
- Public data releases: 14 Hugging Face dataset repos are published with
  `README.md`, `dataset_manifest.json`, and uploaded `data/` files. Combined
  uploaded size is about 79.94GB, including duplicate raw/tokenized releases.
- Korean section: [한국어 설명](#한국어-설명)
- Base model: <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>
- Liquid prompting docs: <https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting>
- Liquid chat template docs: <https://docs.liquid.ai/lfm/key-concepts/chat-template>
- Liquid tool-use docs: <https://docs.liquid.ai/lfm/key-concepts/tool-use>

## Status

Training is in progress. Do not treat this card as a final benchmark report yet.
Stage2 is the main KO-SFT model line. Stage3 Agentic/Fable training is published
separately under `LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT`.

| stage | status | samples | tokens | max seq | note |
|---|---|---:|---:|---:|---|
| Stage0 legal | completed | 8,747 | 35,068,923 | 8192 | Korean legal source/bar-style warmup |
| Stage0b finance/Text2SQL | completed/uploaded | 280,000 | 58,090,087 | 4096 | 8 x H200 full SFT, 2,188 planned steps |
| Stage1 4k finance/Text2SQL | completed/uploaded | 2,302,304 | 1,285,864,494 | 4096 | 8 x H200 full SFT |
| Stage1 8k legal/terminal | running | 1,600,835 | 1,658,848,754 | 8192 | legal long-context and terminal/tool behavior |
| Stage2 diverse KO/SWE/reasoning | prepared | 1,467,864 | 1,364,349,642 | 4096 | excludes raw CPT corpora |
| Stage2 plus KoTSQA | prepared | 1,468,598 | 1,364,863,776 | 4096 | adds KoTSQA train split only |
| Stage3 Agentic/Fable | separate repo | 3,943 | 7,124,298 | 8192 | Fable5/Helio + local doc/log grounded SFT |

Current staged main SFT total is about **4.309577B tokens**:

- Stage1 4k finance/Text2SQL: 1.286B tokens
- Stage1 8k legal/terminal: 1.659B tokens
- Stage2 diverse plus KoTSQA: 1.364864B tokens

ETA from the 2026-06-29 15:57 KST status:

| stage | estimated completion |
|---|---|
| Stage1 8k | 2026-06-29 19:10-19:45 KST; 9840 / 12507 steps, 78.7% |
| Stage2 plus KoTSQA | 2026-06-30 02:30-04:00 KST |
| Stage2 quick gate eval | 2026-06-30 05:30-07:30 KST |
| Stage3 Agentic/Fable SFT | 2026-06-30 morning KST |

## Goal

The goal is to keep LFM2.5 chat, tool-use, and general reasoning behavior while
improving Korean legal, finance, Text2SQL, coding, and exact-answer behavior.

The SFT data follows the LFM ChatML-like template and keeps tool-use examples in
the LFM tool-call style. Liquid's public docs describe this format with structured
conversation roles and tool call delimiters such as `<|tool_call_start|>` and
`<|tool_call_end|>`.

## Data

Main source groups:

- Korean legal tasks, bar-style JSON answers, source-grounded legal agent data,
  and RAG-style legal QA. Legal data includes sources from the legalize-kr
  ecosystem: <https://github.com/legalize-kr>.
- Korean finance/accounting instruction data.
- Text2SQL and structured reasoning data.
- Terminal/tool-use and ToolBench-style conversations.
- Coding/SWE data.
- KoTSQA train split for Korean evidence QA and false-premise correction. The
  test split is kept out for later evaluation:
  <https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0>.
- Korean dataset index reviewed for additional candidates:
  <https://github.com/gyunggyung/LLM-Ko-Datasets>.

Project implementation and runbooks are public at:

- SFT code and docs: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT code and docs: <https://github.com/gyunggyung/LFM25-KO-CPT>

Public dataset releases:

| release | kind | size | source / purpose |
|---|---|---:|---|
| [CPT LFM-style full raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Raw-20260627) | raw LFM text JSONL | 20.54GB | Korean Wiki, finance, legal, legal RAG/bar-answer, terminal/tool traces |
| [CPT LFM-style source shards](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Shards-20260627) | source-separated raw shards | 26.20GB | auditable per-source CPT shards |
| [CPT raw mix before LFM wrapping](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-CPT-Full-Raw-Mix-20260627) | raw JSONL | 4.10GB | pre-conversion CPT mix |
| [SFT Stage0 legal 8k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage0-Legal-LFMChat-8K) | tokenized response-only arrays | 0.16GB | legal source/RAG/bar warmup |
| [SFT Stage0b finance/Text2SQL 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage0B-Finance-Text2SQL-LFMChat-4K) | tokenized response-only arrays | 0.26GB | finance and Text2SQL smoke stage |
| [SFT Stage1 finance/Text2SQL 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Finance-Text2SQL-LFMChat-4K) | tokenized response-only arrays | 5.24GB | main finance/accounting and Text2SQL stage |
| [SFT Stage1 legal/terminal 8k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Legal-Terminal-LFMChat-8K) | tokenized response-only arrays | 6.71GB | legal long-context and terminal/tool traces |
| [SFT Stage2 diverse raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-Raw) | raw LFM chat JSONL | 5.61GB | Korean domain, SWE/coding, reasoning, finance/legal/Text2SQL |
| [SFT Stage2 diverse 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-4K) | tokenized response-only arrays | 5.52GB | Stage2 diverse prepared set |
| [KoTSQA train raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-KoTSQA-Train-LFMChat-Raw) | raw LFM chat JSONL | 0.002GB | KoTSQA v2 train only; test held out |
| [SFT Stage2 plus KoTSQA 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Plus-KoTSQA-LFMChat-4K) | tokenized response-only arrays | 5.52GB | planned Stage2 main KO-SFT training set |
| [Agentic/Fable grounded raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-Raw) | raw LFM chat JSONL | 0.04GB | Fable5/Helio plus local docs/log grounded traces |
| [Agentic/Fable grounded 8k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-8K) | tokenized response-only arrays | 0.05GB | Stage3 Agentic/Fable response-only arrays |
| [Dataset index and sources](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Dataset-Index-and-Sources) | source index | tiny | LLM-Ko-Datasets README/LICENSE snapshot |

The current prepared Stage1 pool is about 2.945B tokens:

- 4k finance/Text2SQL: 1.286B tokens
- 8k legal/terminal: 1.659B tokens

The next Stage2 pool is being prepared from Korean domain SFT, behavior mix,
SWE/coding, reasoning, compact finance/legal, and Text2SQL reinforcement data.
Raw CPT-style corpora such as Korean Wikipedia and raw law text are intentionally
excluded from this SFT phase.

## Quick Sanity Evaluation

This is a small `limit=50` vLLM sanity slice, not a final benchmark.

| task | base `LiquidAI/LFM2.5-8B-A1B` | CPT `LFM2.5-8B-A1B-KO-CPT-FULL` |
|---|---:|---:|
| ARC Challenge acc | 0.2000 | 0.2000 |
| HellaSwag acc | 0.4200 | 0.3800 |
| GSM8K exact match | 0.4600 | 0.2200 |
| IFEval strict prompt acc | 0.1600 | 0.1200 |
| TruthfulQA MC2 acc | 0.5546 | 0.5407 |

The current CPT checkpoint is Korean-knowledge heavy and does not improve this
small English/general sanity slice. The ongoing SFT stages are intended to
recover instruction following, reasoning format, legal/finance QA, tool use, and
coding behavior.

## Training Recipe

- Method: full-parameter supervised fine-tuning, not LoRA.
- Precision: BF16.
- Parallelism: `torchrun` DDP across 8 H200 GPUs.
- Optimizer: fused AdamW.
- Scheduler: cosine with warmup.
- Current Stage0b batch: `per_device_train_batch_size=2`,
  `gradient_accumulation_steps=8`, effective batch `128` sequences/update.
- Checkpoints: every 1000 steps with total limit 2, plus final full model.

The direct DDP trainer is used because a previous Hugging Face `Trainer` attempt
loaded the model but stalled before active GPU training on the second stage.

## Evaluation Plan

We will report base, CPT, and SFT under the same vLLM settings. Planned public
benchmark families:

| area | benchmark / probe | purpose |
|---|---|---|
| Official LFM lineage | IFEval, IFBench, Multi-IF | instruction following preservation |
| Official LFM lineage | MATH500, AIME25 | math/reasoning preservation |
| Official LFM lineage | BFCLv3, BFCLv4 | function/tool calling |
| Official LFM lineage | Tau2 Telecom, Tau2 Retail | agentic task behavior |
| Korean language | Global MMLU Korean, KMMLU | Korean knowledge and MCQA |
| Korean domain | legal/bar/accounting/finance probes | target-domain lift |
| Structured output | Text2SQL and JSON exact extraction | format and exact-answer behavior |

Final scores will be added after the same evaluation matrix has been run on all
comparison models.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful Korean legal and finance assistant."},
    {"role": "user", "content": "대한민국 상법상 이사의 충실의무를 간단히 설명해줘."},
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

## Colab Example

```python
!pip install -U transformers accelerate safetensors

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a precise Korean assistant."},
    {"role": "user", "content": "한국어로 LFM2.5 모델을 사용할 때 chat template을 쓰는 이유를 설명해줘."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
output = model.generate(inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## 한국어 설명

`LFM2.5-8B-A1B-KO-SFT`는 `LFM2.5-8B-A1B-KO-CPT-FULL` 위에 이어서 학습하는
한국어 SFT 모델입니다. 목표는 한국어 법률, 금융, 회계, Text2SQL, 코딩, 터미널
및 툴콜 동작을 강화하면서 기존 LFM2.5의 영어 추론과 도구 사용 능력을 유지하는
것입니다.

현재는 최종 릴리스 전 학습 중입니다. 모델 성능 표는 base, CPT, SFT를 같은
vLLM 평가 설정으로 돌린 뒤 업데이트합니다.

한국어 사용 예시는 위 `Usage`와 `Colab Example`을 참고하면 됩니다.

프로젝트 코드와 실행 문서는 GitHub에 공개되어 있습니다.

- SFT: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT: <https://github.com/gyunggyung/LFM25-KO-CPT>
