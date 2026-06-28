#!/usr/bin/env python3
"""Print HF_TOKEN from a local .env file without executing it."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        return 2

    env_path = Path(sys.argv[1])
    if not env_path.exists():
        return 1

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if not line.startswith("HF_TOKEN="):
            continue
        token = line.split("=", 1)[1].strip().strip('"').strip("'")
        if token:
            print(token)
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
