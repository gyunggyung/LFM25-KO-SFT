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

**Important result:** this Stage2 KO-SFT checkpoint is not an improvement over
KO-CPT on the selected public benchmark matrix. It is published for
reproducibility and failure analysis, not as the recommended checkpoint over
`LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`.

Stage2 is the main KO-SFT model line and has been uploaded to this repository.
Stage3 Agentic/Fable training is a separate follow-up model line under
`LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT`.

The first selected full benchmark run shows that this Stage2 SFT checkpoint is
not a blanket improvement over Base/CPT. It preserves or recovers a few axes,
but it is weak on multiple-choice likelihood-style Korean benchmarks. Treat the
numbers below as a diagnostic snapshot for the Stage2 SFT checkpoint, not as the
final Agentic model report.

| stage | status | samples | tokens | max seq | note |
|---|---|---:|---:|---:|---|
| Stage0 legal | completed | 8,747 | 35,068,923 | 8192 | Korean legal source/bar-style warmup |
| Stage0b finance/Text2SQL | completed/uploaded | 280,000 | 58,090,087 | 4096 | 8 x H200 full SFT, 2,188 planned steps |
| Stage1 4k finance/Text2SQL | completed/uploaded | 2,302,304 | 1,285,864,494 | 4096 | 8 x H200 full SFT |
| Stage1 8k legal/terminal | completed/uploaded | 1,600,835 | 1,658,848,754 | 8192 | legal long-context and terminal/tool behavior |
| Stage2 diverse KO/SWE/reasoning | completed | 1,467,864 | 1,364,349,642 | 4096 | excludes raw CPT corpora |
| Stage2 plus KoTSQA | completed/uploaded | 1,468,598 | 1,364,863,776 | 4096 | main KO-SFT checkpoint; adds KoTSQA train split only |
| Stage3 Agentic/Fable | completed/uploaded in separate repo | 3,943 | 7,124,298 | 8192 | diagnostic only; not a public benchmark improvement |

Current staged main SFT total is about **4.309577B tokens**:

- Stage1 4k finance/Text2SQL: 1.286B tokens
- Stage1 8k legal/terminal: 1.659B tokens
- Stage2 diverse plus KoTSQA: 1.364864B tokens

## Experiment Verdict

| checkpoint | verdict | reason |
|---|---|---|
| KO-CPT | strongest current public benchmark line | broad selected benchmark gains remain better than SFT |
| KO-SFT Stage2 | failed as public benchmark improvement | most IFEval/GSM8K/ARC/PIQA/Korean MCQA axes fell below Base/CPT |
| KO-Agentic Stage3 | failed as public benchmark improvement | small partial recovery only; intended behavior data is not benchmark repair data |

If another SFT experiment is run later, the safer starting point is KO-CPT, not
this regressed KO-SFT checkpoint. The next run should be a small MCQA and
answer-format repair SFT with frequent gates.

## Stage2 Selected Full Benchmark Snapshot

Evaluation was run with vLLM/lm-eval on the uploaded Stage2 full checkpoint.
Base and CPT reference values are copied from the CPT model card for the same
task axes. `KMMLU direct hard STEM` failed once during a crowded vLLM queue and
is marked as pending rather than reported here.

| task | metric | Base | CPT | KO-SFT Stage2 | SFT vs Base | SFT vs CPT |
|---|---|---:|---:|---:|---:|---:|
| IFEval | prompt loose acc | 0.2921 | 0.3216 | 0.1738 | -0.1183 | -0.1478 |
| Leaderboard IFEval | prompt loose acc | 0.2902 | 0.3457 | 0.1756 | -0.1146 | -0.1701 |
| GSM8K | exact match | 0.4845 | 0.5701 | 0.3381 | -0.1464 | -0.2320 |
| BoolQ | acc | 0.6544 | 0.7902 | 0.6664 | +0.0120 | -0.1238 |
| ARC-Challenge | acc_norm | 0.3771 | 0.4241 | 0.2287 | -0.1484 | -0.1954 |
| PIQA | acc_norm | 0.7203 | 0.7476 | 0.5930 | -0.1273 | -0.1546 |
| Global MMLU KO medical genetics | acc | 0.2900 | 0.3800 | 0.3000 | +0.0100 | -0.0800 |
| Global MMLU KO nutrition | acc | 0.2549 | 0.3203 | 0.2157 | -0.0392 | -0.1046 |
| Global MMLU KO philosophy | acc | 0.2669 | 0.3215 | 0.1994 | -0.0675 | -0.1221 |
| Global MMLU KO miscellaneous | acc | 0.3372 | 0.3921 | 0.2401 | -0.0971 | -0.1520 |
| Global MMLU KO professional medicine | acc | 0.3235 | 0.2316 | 0.1838 | -0.1397 | -0.0478 |
| Global MMLU KO high school statistics | acc | 0.2870 | 0.1574 | 0.2222 | -0.0648 | +0.0648 |
| Global MMLU KO astronomy | acc | 0.3421 | 0.2829 | 0.1974 | -0.1447 | -0.0855 |
| Global MMLU KO high school computer science | acc | 0.3100 | 0.2800 | 0.2800 | -0.0300 | +0.0000 |
| Global MMLU KO jurisprudence | acc | 0.2870 | 0.2685 | 0.2593 | -0.0277 | -0.0092 |
| KMMLU direct hard | exact match | 0.2015 | 0.1720 | 0.1055 | -0.0960 | -0.0665 |
| MMLU-ProX Lite KO | exact match | 0.2585 | 0.1667 | 0.0867 | -0.1718 | -0.0800 |

Interpretation:

- Stage2 SFT preserved only a small subset of public benchmark axes. BoolQ is
  slightly above Base, Global MMLU KO medical genetics is slightly above Base,
  and high school statistics recovers part of the CPT regression.
- Korean multiple-choice and exact-answer tasks are mostly below Base/CPT. This
  suggests the SFT mix improved conversation/domain behavior more than
  likelihood-style option selection.
- The next SFT data mix should add explicit Korean MCQA formats: question,
  choices, answer-only labels, and short rationales with the final option
  separated. This is especially important for KMMLU, Global MMLU KO, and
  MMLU-ProX style evaluation.

## Stage3 Agentic/Fable Diagnostic Snapshot

Stage3 Agentic/Fable was trained as a separate model line with Fable5/Helio and
workspace document/log grounding. It was useful as a behavior experiment but did
not repair public benchmark quality.

| task | Stage2 | Agentic/Fable | change |
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

This is not enough to call Stage3 successful. The stage is too small
7.12M tokens, and its data targets terminal/log/document behavior rather than
multiple-choice likelihood or exact-answer repair.

## Failure Analysis

The main failure mode is a mismatch between SFT behavior data and public
benchmark scoring. The Stage2 mix teaches long Korean legal/finance answers,
terminal/tool traces, Text2SQL, coding, and evidence QA. Those are useful
assistant behaviors, but public MCQA benchmarks often score answer-token
likelihood or exact final option extraction. A model can become more verbose and
domain-specific while becoming worse at selecting a short option token.

The response-only SFT format also did not directly optimize the choice ranking
used by KMMLU, Global MMLU KO, and MMLU-ProX. KoTSQA is useful for evidence QA
and false-premise correction, but it is not a direct MCQA repair set. Agentic
Fable data is even further from public benchmark repair: it targets log reading,
tool planning, and grounded terminal behavior.

Next time, the repair experiment should start from KO-CPT and use a compact
100M-300M token set focused on Korean MCQA, answer-only outputs, short
rationales, final-option separation, and strict JSON/exact-answer formats. It
should be stopped immediately if quick gates fall below KO-CPT.

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

The Stage2 pool was prepared from Korean domain SFT, behavior mix, SWE/coding,
reasoning, compact finance/legal, and Text2SQL reinforcement data. Raw CPT-style
corpora such as Korean Wikipedia and raw law text were intentionally excluded
from this SFT phase.

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
small English/general sanity slice. The SFT stages were intended to recover
instruction following, reasoning format, legal/finance QA, tool use, and coding
behavior, but the selected public benchmark results show that this attempt did
not preserve broad benchmark quality.

## Training Recipe

- Method: full-parameter supervised fine-tuning, not LoRA.
- Precision: BF16.
- Parallelism: `torchrun` DDP across 8 H200 GPUs.
- Optimizer: fused AdamW.
- Scheduler: cosine with warmup.
- Stage0b batch: `per_device_train_batch_size=2`,
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

The selected public matrix above is enough to mark the Stage2 KO-SFT line as a
failed public-benchmark improvement. Slower official-card harnesses should be
treated as future optional diagnostics, not as a reason to claim this checkpoint
is stronger than KO-CPT.

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

2026-06-30 기준 공개 벤치 결과는 실패로 판정합니다. Stage2 KO-SFT는 BoolQ와
일부 Global MMLU KO 세부 항목에서만 제한적으로 회복했고, IFEval, GSM8K,
ARC-Challenge, PIQA, KMMLU, MMLU-ProX Lite KO 등 핵심 공개 벤치에서는 Base/CPT
보다 크게 낮았습니다. Stage3 Agentic/Fable도 일부 작은 회복은 있었지만 공개
벤치 개선 모델로 보기에는 부족합니다.

따라서 현재 대표 모델은 KO-CPT입니다. 이 KO-SFT 모델은 재현성과 실패 원인 분석
목적으로 공개합니다. 다시 SFT를 한다면 이 체크포인트에서 이어가는 것보다
KO-CPT에서 작은 다지선다/정확답 repair SFT를 새로 시작하는 편이 낫습니다.

한국어 사용 예시는 위 `Usage`와 `Colab Example`을 참고하면 됩니다.

프로젝트 코드와 실행 문서는 GitHub에 공개되어 있습니다.

- SFT: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT: <https://github.com/gyunggyung/LFM25-KO-CPT>
