#!/usr/bin/env python3
"""Validate raw CPT repair SFT JSONL before tokenization or training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ANSWER_ONLY_RE = re.compile(r"^([0-9]+|[A-Ea-e]|[①②③④⑤])$")
FORBIDDEN_SPLIT_RE = re.compile(r"\b(test|validation|dev)\b", re.IGNORECASE)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle, 1):
            if line.strip():
                yield i, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-answer-only-ratio", type=float, default=0.20)
    parser.add_argument("--max-response-chars", type=int, default=1600)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    stats: dict[str, Any] = {
        "input": str(path),
        "rows": 0,
        "empty_instruction": 0,
        "empty_response": 0,
        "long_response": 0,
        "answer_only_rows": 0,
        "forbidden_split_mentions": 0,
        "by_category": {},
        "warnings": [],
        "errors": [],
    }

    for line_no, row in iter_jsonl(path):
        stats["rows"] += 1
        instruction = str(row.get("instruction") or "").strip()
        response = str(row.get("response") or "").strip()
        category = str(row.get("category") or "unknown")
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        if not instruction:
            stats["empty_instruction"] += 1
            stats["errors"].append(f"line {line_no}: empty instruction")
        if not response:
            stats["empty_response"] += 1
            stats["errors"].append(f"line {line_no}: empty response")
        if len(response) > args.max_response_chars:
            stats["long_response"] += 1
        if ANSWER_ONLY_RE.match(response):
            stats["answer_only_rows"] += 1
        metadata_text = json.dumps(row.get("metadata", {}), ensure_ascii=False)
        source_text = str(row.get("source") or "")
        if FORBIDDEN_SPLIT_RE.search(metadata_text) and '"split": "train"' not in metadata_text:
            stats["forbidden_split_mentions"] += 1
            stats["errors"].append(f"line {line_no}: forbidden split mention in metadata")
        if FORBIDDEN_SPLIT_RE.search(source_text):
            stats["forbidden_split_mentions"] += 1
            stats["warnings"].append(f"line {line_no}: split-like source text")

    if stats["rows"] == 0:
        stats["errors"].append("empty file")
        answer_only_ratio = 0.0
    else:
        answer_only_ratio = stats["answer_only_rows"] / stats["rows"]
    stats["answer_only_ratio"] = answer_only_ratio
    if answer_only_ratio < args.min_answer_only_ratio:
        stats["warnings"].append(
            f"answer_only_ratio {answer_only_ratio:.4f} below {args.min_answer_only_ratio:.4f}"
        )
    if stats["long_response"]:
        stats["warnings"].append(f"{stats['long_response']} responses exceed {args.max_response_chars} chars")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    if stats["errors"] or (args.fail_on_warning and stats["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
