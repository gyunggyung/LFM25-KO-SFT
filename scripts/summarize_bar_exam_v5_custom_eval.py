#!/usr/bin/env python3
"""Summarize custom bar-exam v5 JSON summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_bar_exam_v5_custom_eval.py <out_dir>")
    root = Path(sys.argv[1])
    summaries = []
    for path in sorted(root.glob("*.summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    print("# BarExam V5 Custom Eval Summary")
    print()
    print(f"Root: `{root}`")
    print()
    print("| model | total | correct | accuracy | extracted | extraction_rate | result |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in summaries:
        print(
            f"| `{row['label']}` | {row['total']} | {row['correct']} | "
            f"{row['accuracy']:.6f} | {row['extracted']} | {row['extraction_rate']:.6f} | "
            f"`{row['result_path']}` |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
