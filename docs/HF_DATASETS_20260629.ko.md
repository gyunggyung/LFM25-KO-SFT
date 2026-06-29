# Hugging Face Dataset Releases - 2026-06-29

이 문서는 `LFM2.5-8B-A1B-KO-CPT-FULL`, `LFM2.5-8B-A1B-KO-SFT`,
`LFM2.5-8B-A1B-KO-Agentic-SFT`에 쓰인 공개 데이터 릴리스 현황을 정리한다.

## 상태

최신 확인 기준:

- 14개 dataset repo 생성 완료
- 14개 모두 `README.md`, `dataset_manifest.json`, `data/` 파일 존재
- 합산 업로드 크기: 약 79.94GB
- 포함 토큰 규모: CPT 약 4B tokens, main SFT 약 4.31B tokens, Agentic/Fable 약 7.1M tokens

## 공개 Dataset Repos

| release | size | kind | source / purpose |
|---|---:|---|---|
| [CPT LFM-style full raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Raw-20260627) | 20.54GB | raw LFM text JSONL | Korean Wiki, finance, legal, legal RAG/bar-answer, terminal/tool traces |
| [CPT LFM-style source shards](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Shards-20260627) | 26.20GB | source-separated JSONL | auditable per-source CPT shards |
| [CPT raw mix before LFM wrapping](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-CPT-Full-Raw-Mix-20260627) | 4.10GB | raw JSONL | pre-conversion CPT mix |
| [SFT Stage0 legal 8k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage0-Legal-LFMChat-8K) | 0.16GB | tokenized arrays | legal source/RAG/bar warmup |
| [SFT Stage0b finance/Text2SQL 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage0B-Finance-Text2SQL-LFMChat-4K) | 0.26GB | tokenized arrays | finance and Text2SQL smoke stage |
| [SFT Stage1 finance/Text2SQL 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Finance-Text2SQL-LFMChat-4K) | 5.24GB | tokenized arrays | main finance/accounting and Text2SQL stage |
| [SFT Stage1 legal/terminal 8k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Legal-Terminal-LFMChat-8K) | 6.71GB | tokenized arrays | legal long-context and terminal/tool traces |
| [SFT Stage2 diverse raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-Raw) | 5.61GB | raw LFM chat JSONL | Korean domain, SWE/coding, reasoning, finance/legal/Text2SQL |
| [SFT Stage2 diverse 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-4K) | 5.52GB | tokenized arrays | Stage2 diverse prepared set |
| [KoTSQA train raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-KoTSQA-Train-LFMChat-Raw) | 0.002GB | raw LFM chat JSONL | KoTSQA v2 train split only; test held out |
| [SFT Stage2 plus KoTSQA 4k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Plus-KoTSQA-LFMChat-4K) | 5.52GB | tokenized arrays | planned Stage2 main KO-SFT training set |
| [Agentic/Fable grounded raw](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-Raw) | 0.04GB | raw LFM chat JSONL | Fable5/Helio plus local docs/log grounded traces |
| [Agentic/Fable grounded 8k](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-8K) | 0.05GB | tokenized arrays | Stage3 Agentic/Fable response-only arrays |
| [Dataset index and sources](https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Dataset-Index-and-Sources) | tiny | source index | `gyunggyung/LLM-Ko-Datasets` README/LICENSE snapshot |

## Source Attribution

Main source families:

- Korean Wiki/general knowledge: `kowiki_raw_full_20260524`
- Korean finance/accounting: BCAI finance family and local HRM finance conversions
- Korean legal: Legalize-KR/law.go.kr style legal raw/tasks/RAG/bar-answer data
- Terminal/tool: LFM2.5 Terminal ToolBench style conversations
- Coding/SWE: SWE Zero and GLM/SWE-style repair/reasoning data
- Korean evidence QA: KoTSQA v2.0 train split
- Agentic/Fable: local Fable5 Korean traces, Helio traces, and workspace docs/log grounded examples

Important public links:

- Legalize-KR organization: https://github.com/legalize-kr
- Korean statutes: https://github.com/legalize-kr/legalize-kr
- Korean court precedents: https://github.com/legalize-kr/precedent-kr
- Korean administrative rules: https://github.com/legalize-kr/admrule-kr
- Korean local ordinances: https://github.com/legalize-kr/ordinance-kr
- Legalize-KR pipeline: https://github.com/legalize-kr/legalize-pipeline
- KoTSQA v2.0: https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0
- Korean dataset candidate index: https://github.com/gyunggyung/LLM-Ko-Datasets
- Liquid LFM chat template: https://docs.liquid.ai/lfm/key-concepts/chat-template
- Liquid LFM tool use: https://docs.liquid.ai/lfm/key-concepts/tool-use

## Usage Notes

The raw JSONL releases are useful for audit, reweighting, filtering, or rebuilding
the tokenized arrays. The tokenized releases are response-only SFT artifacts
created with the LFM tokenizer and should be used with the matching training code.

KoTSQA `test` is intentionally not included in the SFT training artifact. It
should remain available for Korean evidence-QA evaluation.

Legal and finance data are training/evaluation artifacts only. They are not
legal, financial, or investment advice. Downstream users should verify upstream
licenses before redistribution or commercial use.
