#!/usr/bin/env python3
"""Upload a completed model folder to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def token_from_env_file(env_file: Path) -> str | None:
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a model folder to HF Hub")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--commit-message", required=True)
    parser.add_argument("--env-file", default="/home/work/.projects/LLM-OS-Models/Terminal/.env")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"folder does not exist: {folder}")
    token = os.environ.get("HF_TOKEN") or token_from_env_file(Path(args.env_file))
    if not token:
        raise SystemExit("HF_TOKEN is required")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=".",
        commit_message=args.commit_message,
    )
    print(f"uploaded folder={folder} repo={args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
