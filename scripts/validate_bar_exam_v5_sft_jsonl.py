#!/usr/bin/env python3
"""Validate bar-exam v5 SFT JSONL before tokenization."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ANSWER_PATTERNS = [
    re.compile(r"정답\s*:\s*([1-5])"),
    re.compile(r"최종\s*답\s*:\s*([1-5])\s*번"),
]
ANSWER_CATEGORIES = {
    "mcqa_safe_answer_symbol",
    "mcqa_safe_answer_numeric",
    "v5_context_grounded_full_solution",
    "v5_context_grounded_answer_compact",
}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if line.strip():
                yield lineno, json.loads(line)


def extract_answer(text: str) -> str | None:
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def validate(path: Path, mode: str) -> dict:
    report = {
        "input": str(path),
        "mode": mode,
        "rows": 0,
        "category_counts": {},
        "source_counts": {},
        "round_counts": {},
        "answer_label_rows": 0,
        "round15_answer_label_rows": 0,
        "errors": [],
        "warnings": [],
    }
    category_counts = Counter()
    source_counts = Counter()
    round_counts = Counter()
    by_category_round = defaultdict(Counter)

    for lineno, row in iter_jsonl(path):
        report["rows"] += 1
        category = str(row.get("category", ""))
        source = str(row.get("source", ""))
        metadata = row.get("metadata") or {}
        round_no = metadata.get("round")
        category_counts[category] += 1
        source_counts[source] += 1
        if round_no is not None:
            round_counts[str(round_no)] += 1
            by_category_round[category][str(round_no)] += 1

        text = row.get("text")
        instruction = row.get("instruction")
        response = row.get("response")
        if text is None and (not instruction or not response):
            report["errors"].append(f"line {lineno}: row must have text or instruction/response")
            continue
        if text is not None and "<|im_start|>assistant" not in str(text):
            report["errors"].append(f"line {lineno}: text row has no assistant turn")

        if category in ANSWER_CATEGORIES:
            report["answer_label_rows"] += 1
            answer_text = str(response if response is not None else text)
            answer = extract_answer(answer_text)
            if answer not in {"1", "2", "3", "4", "5"}:
                report["errors"].append(f"line {lineno}: answer category without normalized 1-5 answer")
            meta_answer = str(metadata.get("answer", "")).strip()
            if meta_answer and meta_answer not in {"1", "2", "3", "4", "5"}:
                report["errors"].append(f"line {lineno}: metadata answer is not normalized 1-5")
            if str(round_no) == "15" or metadata.get("uses_15th_answer_label"):
                report["round15_answer_label_rows"] += 1
                if mode == "holdout_clean":
                    report["errors"].append(f"line {lineno}: holdout_clean contains round-15 answer label")

        if category == "v5_answer_free_procedure":
            payload = str(response if response is not None else text)
            if extract_answer(payload):
                report["errors"].append(f"line {lineno}: answer-free v5 procedure row leaks an answer")

    report["category_counts"] = dict(category_counts)
    report["source_counts"] = dict(source_counts)
    report["round_counts"] = dict(round_counts)
    report["by_category_round"] = {k: dict(v) for k, v in by_category_round.items()}
    if not report["rows"]:
        report["errors"].append("empty input")
    if mode in {"context_solver", "product_tuned"} and not report["round15_answer_label_rows"]:
        report["warnings"].append("context_solver/product_tuned has no round-15 answer-label rows")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mode", choices=["holdout_clean", "context_solver", "product_tuned"], required=True)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()

    report = validate(Path(args.input), args.mode)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["errors"] or (args.fail_on_warning and report["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
