#!/usr/bin/env python3
"""Render a compact Markdown summary from lm-eval output directories."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PREFERRED_METRICS = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "acc,none",
    "acc_norm,none",
    "mc2,none",
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def iter_result_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.json")
        if "samples_" not in p.name and p.name != "config.json"
    )


def pick_metric(metrics: dict[str, Any]) -> tuple[str, Any] | None:
    for key in PREFERRED_METRICS:
        if key in metrics:
            return key, metrics[key]
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not key.endswith("_stderr"):
            return key, value
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_lm_eval_results.py <lm_eval_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    rows: list[tuple[str, str, str, float | str]] = []

    for path in iter_result_files(root):
        data = load_json(path)
        if not data or "results" not in data:
            continue
        model_label = path.parent.name
        for task, metrics in sorted(data["results"].items()):
            if not isinstance(metrics, dict):
                continue
            picked = pick_metric(metrics)
            if picked is None:
                continue
            metric, value = picked
            if isinstance(value, float):
                value = round(value, 6)
            rows.append((model_label, task, metric, value))

    print(f"# lm-eval Summary\n\nRoot: `{root}`\n")
    if not rows:
        print("No lm-eval result rows found yet.")
        return 0

    print("| model | task | metric | value |")
    print("|---|---|---|---:|")
    for model, task, metric, value in rows:
        print(f"| `{model}` | `{task}` | `{metric}` | {value} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
