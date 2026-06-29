#!/usr/bin/env python3
"""Wait until a Transformers model directory is complete and size-stable."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def expected_files(model_dir: Path) -> list[Path]:
    files = [
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
    ]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        files.append(index_path)
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        for name in sorted(set(payload.get("weight_map", {}).values())):
            files.append(model_dir / name)
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
        files.extend(shards)
    return files


def snapshot(files: list[Path]) -> dict[str, int] | None:
    sizes: dict[str, int] = {}
    if not files:
        return None
    for path in files:
        if not path.exists() or not path.is_file():
            return None
        size = path.stat().st_size
        if size <= 0:
            return None
        sizes[str(path)] = size
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for model files to be complete")
    parser.add_argument("model_dir")
    parser.add_argument("--timeout-sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--stable-sec", type=int, default=20)
    parser.add_argument("--checks", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    deadline = time.time() + args.timeout_sec if args.timeout_sec > 0 else None
    previous: dict[str, int] | None = None
    stable_count = 0

    while True:
        if deadline is not None and time.time() > deadline:
            raise SystemExit(f"timeout waiting for complete model dir: {model_dir}")

        if model_dir.is_dir():
            current = snapshot(expected_files(model_dir))
            if current is not None and current == previous:
                stable_count += 1
            else:
                stable_count = 0
            previous = current
            if current is not None and stable_count >= args.checks:
                total = sum(current.values())
                print(f"model_dir_ready={model_dir} files={len(current)} bytes={total}")
                return 0
        time.sleep(args.stable_sec)


if __name__ == "__main__":
    raise SystemExit(main())
