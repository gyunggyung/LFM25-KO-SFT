#!/usr/bin/env python3
"""Upload LFM2 KO SFT raw/prepared datasets to Hugging Face dataset repos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, upload_folder


ROOT = Path("/home/work/.projects/LLM-OS-Models/Terminal")
DATA_ROOT = Path("/home/work/.data/lfm2_ko_sft")
CPT_ROOT = Path("/home/work/.data/lfm2_ko_cpt")
PREP = DATA_ROOT / "prepared" / "lfm_chat"
STAGE = DATA_ROOT / "hf_dataset_upload_stage"


DATASETS = [
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Raw-20260627",
        "kind": "raw_lfm_chat_jsonl",
        "path": CPT_ROOT / "datasets" / "ko_cpt_mix_full_lfmstyle_20260627.jsonl",
        "description": "Full Korean CPT mix converted to LFM-style text JSONL, about 4B-token training source.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Shards-20260627",
        "kind": "raw_lfm_chat_jsonl",
        "path": CPT_ROOT / "datasets" / "shards_full_lfmstyle_20260627",
        "description": "Source-separated LFM-style CPT shards: Korean Wiki, finance, legal raw/tasks/RAG/bar answers, and terminal ToolBench.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-CPT-Full-Raw-Mix-20260627",
        "kind": "raw_jsonl",
        "path": CPT_ROOT / "datasets" / "ko_cpt_mix_full_20260627.jsonl",
        "description": "Full Korean CPT raw mix before LFM-style wrapping.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage0-Legal-LFMChat-8K",
        "kind": "prepared_tokenized",
        "path": PREP / "20260628_lfmchat_stage0_legal",
        "description": "Stage0 Korean legal warmup, LFM tokenizer, response-only SFT arrays.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage0B-Finance-Text2SQL-LFMChat-4K",
        "kind": "prepared_tokenized",
        "path": PREP / "20260628_lfmchat_stage0b_fast_mix_4k_finance_text2sql",
        "description": "Stage0b finance/Text2SQL/legal fast mix, LFM tokenizer.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Finance-Text2SQL-LFMChat-4K",
        "kind": "prepared_tokenized",
        "path": PREP / "20260628_lfmchat_stage1_ko_finance_terminal_text2sql_4k_finance_text2sql",
        "description": "Stage1 4k finance/Text2SQL prepared SFT arrays.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Legal-Terminal-LFMChat-8K",
        "kind": "prepared_tokenized",
        "path": PREP / "20260628_lfmchat_stage1_ko_finance_terminal_text2sql_8k_legal_terminal",
        "description": "Stage1 8k Korean legal/terminal/tool-use prepared SFT arrays.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-Raw",
        "kind": "raw_lfm_chat_jsonl",
        "path": PREP / "20260628_lfmchat_stage2_diverse_ko_swe_reasoning.jsonl",
        "description": "Stage2 raw LFM chat JSONL shards: Korean domain, behavior, SWE/coding, reasoning, finance, legal, Text2SQL.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-4K",
        "kind": "prepared_tokenized",
        "path": PREP / "20260628_lfmchat_stage2_diverse_ko_swe_reasoning_4k",
        "description": "Stage2 diverse Korean/SWE/reasoning prepared SFT arrays, LFM tokenizer.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-KoTSQA-Train-LFMChat-Raw",
        "kind": "raw_lfm_chat_jsonl",
        "path": PREP / "20260628_lfmchat_stage2_plus_kotsqa.jsonl",
        "description": "KoTSQA v2 train split converted to LFM chat JSONL. Test split is held out for evaluation.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Plus-KoTSQA-LFMChat-4K",
        "kind": "prepared_tokenized",
        "path": PREP / "20260628_lfmchat_stage2_plus_kotsqa_4k",
        "description": "Stage2 diverse prepared arrays plus KoTSQA train supplement.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-Raw",
        "kind": "raw_lfm_chat_jsonl",
        "path": PREP / "20260630_lfmchat_agentic_fable_grounded.jsonl",
        "description": "Fable5/Helio Korean agentic traces and local grounded document/log examples converted to LFM chat JSONL.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-8K",
        "kind": "prepared_tokenized",
        "path": PREP / "20260630_lfmchat_agentic_fable_grounded_8k",
        "description": "Agentic/Fable grounded 8k prepared response-only SFT arrays.",
    },
    {
        "repo": "LLM-OS-Models/LFM2.5-KO-Dataset-Index-and-Sources",
        "kind": "source_index",
        "path": DATA_ROOT / "downloads" / "LLM-Ko-Datasets",
        "description": "Snapshot of gyunggyung/LLM-Ko-Datasets README/LICENSE used as a dataset index while building the SFT mix.",
    },
]


REPO_DETAILS = {
    "LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Raw-20260627": {
        "sources": [
            "Korean Wiki/general knowledge: `kowiki_raw_full_20260524`.",
            "Korean finance/accounting text and instruction data: `bcai_finance_kor_hrm_20260524`, `BCAI-Finance-Kor-1862K`.",
            "Korean legal raw/task/RAG/bar-answer sources: `korean_legal_raw_full_20260523`, `korean_legal_tasks_full_20260524`, `korean_admrule_precedent_raw_full_20260524`, `ko_legal_source_agent_sft_20260621`, `ko_legal_rag_agent_sft_round15_v2`, `current_law_bar_json_answer_sft_20260621`.",
            "Terminal/tool-use traces: `lfm25_terminal_toolbench_hrm_turns_v1`.",
        ],
        "notes": [
            "This is the full CPT source after LFM-style wrapping. It is intended for continued pretraining/mid-training, not final chat SFT.",
            "Legal-domain attribution includes the public Legalize-KR ecosystem and the Korean law source ecosystem documented in the model card.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-CPT-Full-LFMStyle-Shards-20260627": {
        "sources": [
            "Source-separated CPT shards for Korean Wiki, Korean finance, Korean legal raw/task/RAG/bar-answer data, and terminal/tool-use traces.",
            "Shard filenames are preserved under `data/` so consumers can inspect or reweight domains independently.",
        ],
        "notes": [
            "Use this repo when you need source-level filtering, deduplication, or domain-ratio reconstruction.",
            "This is the best public artifact for auditing which CPT source files entered the KO-CPT run.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-CPT-Full-Raw-Mix-20260627": {
        "sources": [
            "Same CPT source family as the LFM-style raw release, before final LFM chat/text wrapping.",
            "Includes Korean Wiki/general text, finance, legal, legal RAG/bar-answer, and terminal/tool traces.",
        ],
        "notes": [
            "Useful for debugging the conversion pipeline or rebuilding the CPT mix with a different prompt/text format.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage0-Legal-LFMChat-8K": {
        "sources": [
            "Legal source-grounded SFT: `006_ko_legal_source_agent_sft_20260621.jsonl`.",
            "Legal RAG SFT: `007_ko_legal_rag_agent_sft_round15_v2.jsonl`.",
            "Bar-style JSON answer SFT: `008_current_law_bar_json_answer_sft_20260621.jsonl`.",
        ],
        "notes": [
            "Warmup stage used to validate LFM tokenizer compatibility and response-only labels.",
            "Legal content is connected to Legalize-KR/law.go.kr style sources as documented in the CPT and SFT model cards.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage0B-Finance-Text2SQL-LFMChat-4K": {
        "sources": [
            "Korean finance/accounting instruction data sampled from the BCAI finance family used locally.",
            "Text2SQL clean DuckDB-style structured reasoning data.",
        ],
        "notes": [
            "Fast full-SFT smoke stage for finance/Text2SQL behavior and 8-GPU DDP stability.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Finance-Text2SQL-LFMChat-4K": {
        "sources": [
            "Finance/accounting SFT: `finance_bcai_hrm` prepared split.",
            "Structured Text2SQL/code-like reasoning: `text2sql_core` prepared split.",
        ],
        "notes": [
            "Main 4k SFT stage for Korean finance/accounting explanations and SQL-style exact structured outputs.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage1-Legal-Terminal-LFMChat-8K": {
        "sources": [
            "Korean legal task corpus: `korean_legal_tasks` prepared split.",
            "Terminal/tool behavior: `terminal_toolbench` prepared split, derived from LFM2.5 Terminal ToolBench style conversations.",
        ],
        "notes": [
            "8k context is used to preserve longer legal and terminal/tool traces.",
            "This stage is full-parameter SFT, not LoRA.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-Raw": {
        "sources": [
            "Korean domain core.",
            "Behavior core covering terminal/tool/SWE/reasoning/legal/finance patterns.",
            "SWE/coding sources: `swe_zero`, `swe_glm_mix`, compact SWE supplement.",
            "Reasoning/agent sources: GLM reasoning, HF extra reasoning/agent mix, compact agent-reasoning supplement.",
            "Compact finance/legal reinforcement and Text2SQL DuckDB data.",
        ],
        "notes": [
            "Raw LFM chat JSONL for Stage2 before tokenization.",
            "CPT-style raw corpora such as Korean Wiki and raw statutes are intentionally excluded from this SFT stage.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Diverse-KoSWE-Reasoning-LFMChat-4K": {
        "sources": [
            "Tokenized version of the Stage2 diverse KO/SWE/reasoning raw LFM chat mix.",
        ],
        "notes": [
            "Prepared with the LFM tokenizer and response-only labels for 4k full SFT.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-KoTSQA-Train-LFMChat-Raw": {
        "sources": [
            "KoTSQA v2.0: https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0.",
            "Only the train split is converted to LFM chat JSONL.",
        ],
        "notes": [
            "The test split is intentionally held out for Korean evidence-QA evaluation.",
            "Used to strengthen Korean table/document-grounded QA and false-premise correction behavior.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-SFT-Stage2-Plus-KoTSQA-LFMChat-4K": {
        "sources": [
            "Stage2 diverse KO/SWE/reasoning prepared set.",
            "KoTSQA v2.0 train split supplement.",
        ],
        "notes": [
            "This is the planned Stage2 main KO-SFT training set.",
            "KoTSQA test split is not included.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-Raw": {
        "sources": [
            "Local Fable5 Korean traces: `fable_distillation/datasets_ko/fable5_ko_sft_20260624.jsonl`.",
            "Local Helio Korean reasoning traces: `fable_distillation/datasets_ko/helio_ko_sft_20260628.jsonl`.",
            "Grounded examples generated from this project's README, runbook, train logs, git-push failure patterns, and vLLM/agent harness docs.",
        ],
        "notes": [
            "This is a separate Stage3 Agentic/Fable follow-up dataset, not part of the main Stage2 KO-SFT release.",
            "The goal is command-following, log inspection, evidence-grounded diagnosis, and terminal/tool style behavior.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-8K": {
        "sources": [
            "Tokenized response-only 8k version of the Agentic/Fable grounded LFM chat raw dataset.",
        ],
        "notes": [
            "Target model repo: `LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT`.",
            "Stage3 is intentionally separated from the main Stage2 SFT model line.",
        ],
    },
    "LLM-OS-Models/LFM2.5-KO-Dataset-Index-and-Sources": {
        "sources": [
            "Snapshot of https://github.com/gyunggyung/LLM-Ko-Datasets README/LICENSE.",
        ],
        "notes": [
            "This is an index/reference artifact, not a training corpus by itself.",
            "It documents Korean dataset candidates reviewed while building the KO SFT mix.",
        ],
    },
}


def load_env_token() -> str | None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            if stripped.startswith(("HF_TOKEN=", "HUGGINGFACE_HUB_TOKEN=")):
                return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def stats_for(path: Path) -> dict:
    stats = {"path": str(path), "size_bytes": dir_size(path)}
    for name in ("merge_stats.json", "preprocess_stats.json"):
        p = path / name
        if p.exists():
            try:
                stats[name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                stats[name] = p.read_text(encoding="utf-8", errors="replace")[:2000]
    if path.is_file():
        stats["file_count"] = 1
        stats["file_name"] = path.name
    if path.is_dir():
        jsonls = sorted(path.glob("*.jsonl"))
        if jsonls:
            stats["jsonl_files"] = [{"name": p.name, "size_bytes": p.stat().st_size} for p in jsonls]
        stats["file_count"] = sum(1 for p in path.rglob("*") if p.is_file())
    return stats


def readme(entry: dict, stats: dict) -> str:
    details = REPO_DETAILS.get(entry["repo"], {})
    source_lines = "\n".join(f"- {line}" for line in details.get("sources", [])) or "- See the manifest below."
    note_lines = "\n".join(f"- {line}" for line in details.get("notes", [])) or "- No additional notes."
    merge = stats.get("merge_stats.json") or stats.get("preprocess_stats.json") or {}
    sample_count = merge.get("samples") or merge.get("sample_count") or merge.get("kept_rows") or "n/a"
    token_count = merge.get("tokens") or merge.get("total_tokens") or stats.get("metadata.json", {}).get("total_length") or "n/a"
    max_seq = merge.get("max_sample_len") or merge.get("max_written_len") or stats.get("metadata.json", {}).get("max_seq_len") or "n/a"
    return f"""---
license: other
language:
- ko
- en
tags:
- lfm
- korean
- sft
- lfm-chat
- {"tokenized" if entry["kind"] == "prepared_tokenized" else "jsonl"}
---

# {entry['repo'].split('/')[-1]}

{entry['description']}

This dataset is part of the `LFM2.5-8B-A1B-KO-SFT` / Agentic SFT workflow.

- Main SFT model: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT
- CPT base model: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL
- Agentic follow-up model: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT
- SFT GitHub: https://github.com/gyunggyung/LFM25-KO-SFT
- CPT GitHub: https://github.com/gyunggyung/LFM25-KO-CPT

## Source Attribution

{source_lines}

Additional public references:

- Liquid LFM base model: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
- Liquid chat template docs: https://docs.liquid.ai/lfm/key-concepts/chat-template
- Liquid tool-use docs: https://docs.liquid.ai/lfm/key-concepts/tool-use
- Legalize-KR organization: https://github.com/legalize-kr
- KoTSQA v2.0: https://huggingface.co/datasets/etri-lirs/KoTSQA-v.2.0
- Korean dataset index reviewed for candidates: https://github.com/gyunggyung/LLM-Ko-Datasets

## Notes

{note_lines}

## Summary

| field | value |
|---|---:|
| kind | `{entry['kind']}` |
| sample count | {sample_count} |
| token count | {token_count} |
| max sequence / sample length | {max_seq} |
| uploaded size bytes | {stats['size_bytes']} |

## Format

- `raw_lfm_chat_jsonl`: JSONL rows with a `text` field containing LFM ChatML-like conversation text.
- `prepared_tokenized`: NumPy response-only SFT arrays built with the LFM tokenizer:
  - `tokens.npy`
  - `epoch_0/inst_start.npy`
  - `epoch_0/inst_len.npy`
  - `epoch_0/resp_start.npy`
  - `epoch_0/resp_len.npy`
  - `tokenizer.json`

## Local Source Path

```text
{stats['path']}
```

## License And Usage Notes

This release republishes preprocessing artifacts used for the LFM2.5 Korean CPT/SFT
workflow. Source components come from multiple public or locally prepared
datasets, so downstream users should verify each upstream source license before
redistribution or commercial use. Legal and finance examples are for model
training/evaluation only and are not legal, financial, or investment advice.

## Stats

```json
{json.dumps(stats, ensure_ascii=False, indent=2)[:12000]}
```
"""


def copy_or_stage(entry: dict, stage_dir: Path) -> Path:
    src = Path(entry["path"])
    if not src.exists():
        raise FileNotFoundError(src)
    repo_stage = stage_dir / entry["repo"].split("/")[-1]
    if repo_stage.exists():
        import shutil

        shutil.rmtree(repo_stage)
    repo_stage.mkdir(parents=True)

    stats = stats_for(src)
    manifest_entry = dict(entry)
    manifest_entry["path"] = str(manifest_entry["path"])
    (repo_stage / "README.md").write_text(readme(entry, stats), encoding="utf-8")
    (repo_stage / "dataset_manifest.json").write_text(
        json.dumps({"entry": manifest_entry, "stats": stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Upload the source folder contents directly to preserve relative names.
    # The staging folder keeps README/manifest; source is uploaded separately.
    return repo_stage


def upload_source(api: HfApi, entry: dict, src: Path, token: str | None) -> None:
    if src.is_dir():
        upload_folder(
            repo_id=entry["repo"],
            repo_type="dataset",
            folder_path=str(src),
            path_in_repo="data",
            token=token,
            commit_message="Upload dataset files",
        )
        return

    from huggingface_hub import upload_file

    upload_file(
        path_or_fileobj=str(src),
        path_in_repo=f"data/{src.name}",
        repo_id=entry["repo"],
        repo_type="dataset",
        token=token,
        commit_message="Upload dataset file",
    )


def selected_entries(names: set[str] | None) -> list[dict]:
    if not names:
        return DATASETS
    out = []
    for entry in DATASETS:
        short = entry["repo"].split("/")[-1]
        if entry["repo"] in names or short in names:
            out.append(entry)
    if not out:
        raise SystemExit(f"No dataset matched: {sorted(names)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="Repo ids or short names to upload.")
    parser.add_argument("--cards-only", action="store_true", help="Create repos and upload README/manifest only.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = load_env_token()
    api = HfApi(token=token)
    STAGE.mkdir(parents=True, exist_ok=True)

    for entry in selected_entries(set(args.only or []) or None):
        src = Path(entry["path"])
        stats = stats_for(src)
        print(json.dumps({"repo": entry["repo"], "path": str(src), "size_gb": round(stats["size_bytes"] / 1e9, 3)}, ensure_ascii=False), flush=True)
        if args.dry_run:
            continue
        repo_stage = copy_or_stage(entry, STAGE)
        api.create_repo(entry["repo"], repo_type="dataset", exist_ok=True)
        upload_folder(
            repo_id=entry["repo"],
            repo_type="dataset",
            folder_path=str(repo_stage),
            path_in_repo=".",
            token=token,
            commit_message="Add dataset card and manifest",
        )
        if args.cards_only:
            print(f"uploaded_card={entry['repo']}", flush=True)
            continue
        upload_source(api, entry, src, token)
        print(f"uploaded={entry['repo']}", flush=True)


if __name__ == "__main__":
    main()
