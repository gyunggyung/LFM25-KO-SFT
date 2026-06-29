#!/usr/bin/env python3
"""Create the Agentic SFT model repo and upload a placeholder model card."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, upload_file


ROOT = Path("/home/work/.projects/LLM-OS-Models/Terminal")
REPO_ID = "LLM-OS-Models/LFM2.5-8B-A1B-KO-Agentic-SFT"


def load_env_token() -> str | None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if stripped.startswith(("HF_TOKEN=", "HUGGINGFACE_HUB_TOKEN=")):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    token = load_env_token()
    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    card = """---
base_model:
- LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT
- LiquidAI/LFM2.5-8B-A1B
license: other
language:
- ko
- en
tags:
- lfm
- korean
- agentic
- fable
- tool-use
- terminal
- sft
pipeline_tag: text-generation
---

# LFM2.5-8B-A1B-KO-Agentic-SFT

Agentic/Fable-style follow-up SFT for `LFM2.5-8B-A1B-KO-SFT`.

- GitHub: <https://github.com/gyunggyung/LFM25-KO-SFT>
- CPT GitHub: <https://github.com/gyunggyung/LFM25-KO-CPT>
- Main KO-SFT model: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT>
- CPT base model: <https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL>
- Agentic/Fable raw data: <https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-Raw>
- Agentic/Fable tokenized data: <https://huggingface.co/datasets/LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-8K>

This repository is reserved for the Stage3 model. Stage1/Stage2 Korean SFT
checkpoints are published under:

<https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT>

## Planned Training

- Method: full-parameter SFT, not RLVR/GRPO for the June 30 deadline.
- Base checkpoint: Stage2 final from `LFM2.5-8B-A1B-KO-SFT`.
- Data: Fable5 Korean agentic traces, Helio Korean reasoning traces, and local
  document/log grounded examples.
- Prepared data: `LLM-OS-Models/LFM2.5-KO-Agentic-Fable-Grounded-LFMChat-8K`.
- Goal: Korean document-grounded reasoning, log diagnosis, terminal/tool-use
  planning, and safe code-assistant behavior.

## Status

The model weights will be uploaded automatically after Stage3 training finishes.
Do not treat this placeholder as a trained checkpoint.
"""
    tmp = Path("/tmp/LFM2.5-8B-A1B-KO-Agentic-SFT-README.md")
    tmp.write_text(card, encoding="utf-8")
    upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        token=token,
        commit_message="Add initial Agentic SFT model card",
    )
    print(f"created_or_updated={REPO_ID}")


if __name__ == "__main__":
    main()
