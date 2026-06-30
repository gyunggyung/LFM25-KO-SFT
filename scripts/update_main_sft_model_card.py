#!/usr/bin/env python3
"""Upload the main KO-SFT model card README."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path("/home/work/.projects/LLM-OS-Models/Terminal/.env")
REPO_ID = "LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT"


def token_from_env_file(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line.startswith("HF_TOKEN="):
            continue
        token = line.split("=", 1)[1].strip().strip('"').strip("'")
        if token:
            return token
    return None


def main() -> int:
    token = os.environ.get("HF_TOKEN") or token_from_env_file(ENV_FILE)
    if not token:
        raise SystemExit("HF_TOKEN is required")
    card_path = ROOT / "model_card.md"
    api = HfApi(token=token)
    api.upload_file(
        repo_id=REPO_ID,
        repo_type="model",
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        commit_message="Update final KO-SFT closeout card",
    )
    print(f"updated README.md for {REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
