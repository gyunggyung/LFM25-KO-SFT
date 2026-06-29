# LFM2.5-8B-A1B-KO-SFT Workspace

This workspace prepares, trains, evaluates, and publishes
`LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT`.

- Korean guide: [`docs/SFT_DATA_PLAN_20260628.ko.md`](docs/SFT_DATA_PLAN_20260628.ko.md)
- Runbook: [`docs/RUNBOOK_20260628.ko.md`](docs/RUNBOOK_20260628.ko.md)
- Final evaluation plan: [`docs/EVAL_PLAN_FINAL_SFT_20260628.ko.md`](docs/EVAL_PLAN_FINAL_SFT_20260628.ko.md)
- External evaluation harnesses: [`docs/EXTERNAL_HARNESS_SETUP_20260628.ko.md`](docs/EXTERNAL_HARNESS_SETUP_20260628.ko.md)
- Agent harness: [`docs/AGENT_HARNESS_20260629.ko.md`](docs/AGENT_HARNESS_20260629.ko.md)
- Agentic/Fable follow-up chain: [`docs/AGENTIC_FABLE_CHAIN_20260630.ko.md`](docs/AGENTIC_FABLE_CHAIN_20260630.ko.md)
- Public Hugging Face datasets: [`docs/HF_DATASETS_20260629.ko.md`](docs/HF_DATASETS_20260629.ko.md)
- Hugging Face model card source: [`model_card.md`](model_card.md)
- Target Hub repo: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT>
- Base model: <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>

## Current Status

| item | status | path / note |
|---|---|---|
| Stage0 legal full SFT | done | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0-legal-20260628/final_full` |
| Stage0b finance/Text2SQL full SFT | done/uploaded | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0b-finance-text2sql-20260628/final_full` |
| Stage1 4k finance/Text2SQL full SFT | done/uploaded | 2,302,304 samples, 1.286B tokens |
| Stage1 8k legal/terminal prepared set | ready | 1,600,835 samples, 1.659B tokens |
| Stage2 diverse KO/SWE/reasoning prepared set | ready | 1.364B tokens, excludes raw CPT-style corpora |
| Stage2 plus KoTSQA | ready | 1,468,598 samples, 1.364864B tokens; adds `etri-lirs/KoTSQA-v.2.0` train split only |
| Main SFT token total | staged | 1.286B + 1.659B + 1.364864B = 4.309577B |
| Stage3 Agentic/Fable SFT | prepared by chain | Fable5 KO + Helio KO + local docs/logs grounding, 8k context |
| Evaluation results | partial | quick base/CPT sanity slice exists; SFT eval deferred to keep GPUs training |
| Public HF datasets | uploaded | 14 dataset repos, all with `data/`, README, and manifest; about 79.94GB uploaded |

The active run uses full-parameter SFT, not LoRA. The working launcher is
the direct `torchrun` DDP path because the earlier `Trainer` run loaded weights but
stalled with near-zero GPU memory use during the second stage.

## Why LFM-Style Preprocessing

The first prepared SFT datasets were not safe for LFM because they contained token
ids above the LFM model vocabulary range. That caused CUDA device-side asserts
during training. The fixed pipeline rebuilds samples with the LFM tokenizer and
stores response-only labels in the format expected by `scripts/train_lfm25_ko_sft_torchrun.py`.

Liquid's public docs define the relevant behavior we preserve:

- Prompting roles: `system`, `user`, and `assistant`
  (<https://docs.liquid.ai/lfm/key-concepts/text-generation-and-prompting>)
- ChatML-like LFM chat template with `system`, `user`, `assistant`, and `tool`
  roles (<https://docs.liquid.ai/lfm/key-concepts/chat-template>)
- Tool calls wrapped with `<|tool_call_start|>` and `<|tool_call_end|>`
  (<https://docs.liquid.ai/lfm/key-concepts/tool-use>)

## Training Commands

Stage0 legal was completed with:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_prepare_lfmchat_stage0_legal.sh
bash scripts/run_lfm25_ko_sft_stage0_legal.sh
```

The current Stage1 4k run is:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
tmux attach -t lfm2ko_sft_stage1_4k_20260628
```

Equivalent direct command:

```bash
DATASET_PATH=/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage1_ko_finance_terminal_text2sql_4k_finance_text2sql \
MODEL_PATH=/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0b-finance-text2sql-20260628/final_full \
RUN_ID=stage1-4k-finance-text2sql-20260628 \
STAGE_NAME=stage1_4k_finance_text2sql \
MAX_SEQ_LENGTH=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=2 \
PUSH_TO_HUB=1 \
bash scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh
```

Effective batch size is `8 GPUs * 2 sequences/GPU * 8 accumulation = 128`
sequences/update. Stage1 4k has 2,302,304 samples and is planned for 17,987
update steps.

## Prepared Data

Prepared data is stored under:

```text
/home/work/.data/lfm2_ko_sft/prepared/lfm_chat
```

Important prepared sets:

| split | path | max seq | samples | tokens |
|---|---|---:|---:|---:|
| Stage0 legal | `20260628_lfmchat_stage0_legal` | 8192 | 8,747 | 35,068,923 |
| Stage0b fast mix | `20260628_lfmchat_stage0b_fast_mix_4k_finance_text2sql` | 4096 | 280,000 | 58,090,087 |
| Stage1 4k finance/Text2SQL | `20260628_lfmchat_stage1_ko_finance_terminal_text2sql_4k_finance_text2sql` | 4096 | 2,302,304 | 1,285,864,494 |
| Stage1 8k legal/terminal | `20260628_lfmchat_stage1_ko_finance_terminal_text2sql_8k_legal_terminal` | 8192 | 1,600,835 | 1,658,848,754 |
| Stage2 diverse KO/SWE/reasoning | `20260628_lfmchat_stage2_diverse_ko_swe_reasoning_4k` | 4096 | 1,467,864 | 1,364,349,642 |
| Stage2 plus KoTSQA | `20260628_lfmchat_stage2_plus_kotsqa_4k` | 4096 | 1,468,598 | 1,364,863,776 |

See the data plan for source URLs, local paths, and ratios.

KoTSQA is being prepared as a Stage2 supplement under:

```text
/home/work/.data/lfm2_ko_sft/prepared/lfm_chat/20260628_lfmchat_stage2_plus_kotsqa_4k
```

It uses the `train` split from <https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0>
only. The `test` split is intentionally held out for later Korean QA evaluation.

## Training Order

1. Stage1 4k finance/Text2SQL on 8 GPUs.
2. Stage1 8k legal/terminal for legal long-context and tool behavior.
3. Stage2 4k diverse KO/SWE/reasoning plus KoTSQA.
4. Run the Stage2 quick vLLM gate comparison.
5. Run Agentic/Fable SFT for document/log grounded terminal behavior.
6. Run the agentic quick vLLM comparison and agent harness smoke.
7. Update the model cards and expand official-card harnesses.

Priority rule: keep the full SFT chain running first. CPU/network setup, docs,
and harness installation can run in parallel, but GPU evaluation waits until a
training stage releases GPUs unless it is a deliberate quick gate.

Automatic chain sessions:

```bash
tmux attach -t lfm2ko_chain_after_stage1_20260628
tmux attach -t lfm2ko_chain_stage2_after_8k_20260628
tmux attach -t lfm2ko_chain_agentic_after_stage2_20260630
tmux attach -t lfm2ko_setup_external_harnesses_20260628
```

ETA refreshed from the 2026-06-29 15:05 KST status:

| item | tokens | estimate |
|---|---:|---:|
| Stage1 8k train | 1.659B | 2026-06-29 19:15-19:45 KST |
| Stage2 diverse plus KoTSQA train | 1.364864B | 2026-06-30 02:30-04:00 KST |
| Stage2 quick gate eval | limit 50 | 2026-06-30 05:30-07:30 KST |
| Agentic/Fable SFT | Fable/log/doc SFT | 2026-06-30 08:00-12:00 KST |
| Agentic quick eval + smoke | limit 50 + harness | 2026-06-30 13:00-15:30 KST |

These windows are estimates and should be refreshed from `train_log.jsonl` after
each stage starts.

The follow-up chain intentionally skips GRPO/RLVR for the June 30 deadline.
Agentic behavior is trained with SFT traces first; RLVR should be a later
experiment only after tool success metrics and reward checks are stable.

## Agentic/Fable Follow-Up

The June 30 follow-up makes the model behave more like Fable-style terminal
agents: read evidence, inspect logs, choose safe commands, explain the cause,
and verify results. It is SFT-only.

Prepared sources:

- Fable5 Korean traces:
  `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets_ko/fable5_ko_sft_20260624.jsonl`
- Helio Korean reasoning traces:
  `/home/work/.projects/LLM-OS-Models/Terminal/fable_distillation/datasets_ko/helio_ko_sft_20260628.jsonl`
- Local grounded examples generated from this workspace's README, runbook,
  train logs, git push error patterns, and vLLM/agent harness docs.

Preparation:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_prepare_lfmchat_agentic_fable_grounded.sh
```

Automatic Stage2 -> gate eval -> Agentic SFT -> agentic eval chain:

```bash
tmux new -d -s lfm2ko_chain_agentic_after_stage2_20260630 \
  'cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft && bash scripts/run_agentic_fable_chain_after_stage2.sh'
```

Details: [`docs/AGENTIC_FABLE_CHAIN_20260630.ko.md`](docs/AGENTIC_FABLE_CHAIN_20260630.ko.md).

## Quick Eval Snapshot

The 2026-06-28 quick vLLM sanity run used `limit=50`, so it is not a final
benchmark. It was used only to catch broad regressions before continuing SFT.

| task | base `LiquidAI/LFM2.5-8B-A1B` | CPT `LFM2.5-8B-A1B-KO-CPT-FULL` |
|---|---:|---:|
| ARC Challenge acc | 0.2000 | 0.2000 |
| HellaSwag acc | 0.4200 | 0.3800 |
| GSM8K exact match | 0.4600 | 0.2200 |
| IFEval strict prompt acc | 0.1600 | 0.1200 |
| TruthfulQA MC2 acc | 0.5546 | 0.5407 |

CPT is Korean-knowledge heavy and does not improve these small English/general
sanity slices. The SFT stages are intended to recover instruction following,
reasoning format, legal/finance QA, tool use, and coding behavior.

## Evaluation Plan

Evaluation should run with vLLM and compare:

1. `LiquidAI/LFM2.5-8B-A1B`
2. `LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL`
3. `LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT`

The official LFM2.5 model card reports these benchmark families for the base
lineage: AA-Omniscience, IFEval, IFBench, Multi-IF, MATH500, AIME25, BFCLv3,
BFCLv4, Tau2 Telecom, and Tau2 Retail. The KO SFT evaluation should add Korean
benchmarks and task probes:

- Global MMLU Korean / KMMLU
- Korean legal MCQA and bar-style JSON answer extraction
- Korean finance/accounting QA and MCQA
- Text2SQL exact-match probes
- Tool-call and terminal behavior probes using the LFM tool-call format

No final SFT score should be claimed until the same prompts and decoding settings
are run for base, CPT, and SFT.

Evaluation scripts:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft

# Single model/task group on one visible GPU.
CUDA_VISIBLE_DEVICES=0 \
MODEL_ID=LiquidAI/LFM2.5-8B-A1B \
TASKS=ifeval,gsm8k \
bash scripts/run_vllm_lm_eval_matrix.sh

# Queue configured models and task groups across up to 8 one-GPU workers.
# Run this only after the active training job releases GPUs.
bash scripts/run_vllm_eval_8gpu_queue.sh
```

The default queue config lives in:

- `configs/eval_models_20260628.txt`
- `configs/eval_task_groups_20260628.txt`

The final post-training queue config lives in:

- `configs/eval_models_final_sft_20260628.txt`
- `configs/eval_task_groups_final_sft_lm_eval_20260628.txt`

It includes LFM official-card overlap available through local lm-eval
(`ifeval`, `minerva_math500`, `aime25`) and Korean probes
(`global_mmlu_ko`, `global_mmlu_full_ko`, `kmmlu_direct_hard`,
`kmmlu_cot_hard`, `kobest`, `haerae`, `belebele_kor_Hang`). Official LFM
items that need separate harnesses are tracked in
`docs/EVAL_PLAN_FINAL_SFT_20260628.ko.md`.

## External Harnesses

Separate harness setup covers the remaining official-card families:

- BFCLv3/BFCLv4 for tool/function calling
- Tau2 Telecom/Retail for agentic tool use
- IFBench and Multi-IF for instruction following
- AA-Omniscience through LightEval once the installed task string is confirmed

Setup can run while training continues:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/setup_external_eval_harnesses.sh
```

The detailed command sheet is
[`docs/EXTERNAL_HARNESS_SETUP_20260628.ko.md`](docs/EXTERNAL_HARNESS_SETUP_20260628.ko.md).

## Agent Harness

The bounded agent harness lives in `agent_harness/`. It is designed for the
tasks the SFT mix should support well: Korean legal/finance QA, terminal/tool
status work, Text2SQL, small code-assistant tasks, and Korean instruction
following.

GPU-free smoke test:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_lfm2ko_agent_smoke_eval.sh
```

Final-model endpoint test after training/evaluation:

```bash
OPENAI_BASE_URL=http://localhost:1053/v1 \
OPENAI_API_KEY=EMPTY \
MODEL_NAME=lfm2-ko-sft \
bash scripts/run_lfm2ko_agent_harness.sh \
  "README.md를 읽고 이 모델의 학습/평가 실행법을 한국어로 요약해라."
```

Details: [`docs/AGENT_HARNESS_20260629.ko.md`](docs/AGENT_HARNESS_20260629.ko.md).

The same harness also supports CPU GGUF through llama.cpp:

```bash
MODEL_GGUF=/path/to/LFM2.5-8B-A1B-KO-SFT-Q8_0.gguf \
CTX_SIZE=8192 \
PORT=8080 \
bash scripts/start_llamacpp_gguf_server_for_agent.sh

AGENT_BACKEND=llamacpp \
OPENAI_BASE_URL=http://localhost:8080/v1 \
MODEL_NAME=lfm2-ko-sft-gguf \
bash scripts/run_lfm2ko_agent_harness.sh \
  --context-window 8192 \
  --prompt-budget 20000 \
  "README.md를 요약해라."
```

## Colab / Inference Example

```python
!pip install -U transformers accelerate safetensors

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "You are a helpful Korean legal and finance assistant."},
    {"role": "user", "content": "대한민국 상법상 이사의 충실의무를 핵심만 설명해줘."},
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
