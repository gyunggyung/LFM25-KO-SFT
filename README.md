# LFM2.5-8B-A1B-KO-SFT Workspace

This workspace prepares, trains, evaluates, and publishes
`LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT`.

- Korean guide: [`docs/SFT_DATA_PLAN_20260628.ko.md`](docs/SFT_DATA_PLAN_20260628.ko.md)
- Hugging Face model card source: [`model_card.md`](model_card.md)
- Target Hub repo: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT>
- Base model: <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>

## Current Status

| item | status | path / note |
|---|---|---|
| Stage0 legal full SFT | done | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0-legal-20260628/final_full` |
| Stage0b finance/Text2SQL full SFT | done/uploaded | `/home/work/.data/lfm2_ko_sft/models/LFM2.5-8B-A1B-KO-SFT-stage0b-finance-text2sql-20260628/final_full` |
| Stage1 4k finance/Text2SQL full SFT | running on 8 x H200 | 2,302,304 samples, 1.286B tokens, 17,987 planned steps |
| Stage1 8k legal/terminal prepared set | ready | 1,600,835 samples, 1.659B tokens |
| Stage2 diverse KO/SWE/reasoning prepared set | ready | 1.364B tokens, excludes raw CPT-style corpora |
| Stage2 plus KoTSQA | preparing on CPU | adds `etri-lirs/KoTSQA-v.2.0` train split only |
| SFT token total | staged | 1.286B + 1.659B + 1.364B = 4.309B before KoTSQA |
| Evaluation results | partial | quick base/CPT sanity slice exists; SFT eval deferred to keep GPUs training |

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
3. Stage2 4k diverse KO/SWE/reasoning plus KoTSQA if prep completes.
4. Run the full vLLM comparison table and update the model card.

Automatic chain sessions:

```bash
tmux attach -t lfm2ko_chain_after_stage1_20260628
tmux attach -t lfm2ko_chain_stage2_after_8k_20260628
tmux attach -t lfm2ko_prep_kotsqa_stage2_plus_20260628
```

ETA from the 2026-06-28 16:41 KST status:

| item | tokens | estimate |
|---|---:|---:|
| Stage1 4k train | 1.286B | 2026-06-29 03:15-03:45 KST |
| Stage1 8k train | 1.659B | 2026-06-29 19:30-2026-06-30 01:30 KST |
| Stage2 diverse plus KoTSQA train | 1.364B + KoTSQA train | 2026-06-30 07:30-14:00 KST |

These windows are estimates and should be refreshed from `train_log.jsonl` after
each stage starts.

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
