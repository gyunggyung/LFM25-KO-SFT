#!/usr/bin/env python3
"""Create a Base/CPT/SFT comparison table for public benchmark reporting."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "configs" / "benchmark_reference_cpt_card_20260630.json"

MODEL_ALIASES = {
    "LiquidAI/LFM2.5-8B-A1B": "Base",
    "LLM-OS-Models/LFM2.5-8B-A1B-KO-CPT-FULL": "CPT",
}

METRIC_PRIORITY = (
    "prompt_level_loose_acc,none",
    "prompt_level_strict_acc,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match,custom-extract",
    "exact_match,get-answer",
    "exact_match,none",
    "acc_norm,none",
    "acc,none",
    "mc2,none",
)

TASK_LABELS = {
    "ifeval": "IFEval",
    "leaderboard_ifeval": "Leaderboard IFEval",
    "gsm8k": "GSM8K",
    "boolq": "BoolQ",
    "arc_challenge": "ARC-Challenge",
    "piqa": "PIQA",
    "global_mmlu_full_ko_medical_genetics": "Global MMLU KO: medical genetics",
    "global_mmlu_full_ko_nutrition": "Global MMLU KO: nutrition",
    "global_mmlu_full_ko_philosophy": "Global MMLU KO: philosophy",
    "global_mmlu_full_ko_miscellaneous": "Global MMLU KO: miscellaneous",
    "global_mmlu_full_ko_professional_medicine": "Global MMLU KO: professional medicine",
    "global_mmlu_full_ko_high_school_statistics": "Global MMLU KO: high school statistics",
    "global_mmlu_full_ko_astronomy": "Global MMLU KO: astronomy",
    "global_mmlu_full_ko_high_school_computer_science": "Global MMLU KO: high school computer science",
    "global_mmlu_full_ko_jurisprudence": "Global MMLU KO: jurisprudence",
    "kmmlu_direct_hard": "KMMLU direct hard",
    "kmmlu_direct_hard_stem": "KMMLU direct hard STEM",
    "mmlu_prox_lite_ko": "MMLU-ProX Lite KO",
    "mmlu_pro_law": "MMLU-Pro law",
    "mmlu_pro_economics": "MMLU-Pro economics",
}

SECTION_ORDER = (
    (
        "Preserve general and official-card strengths",
        (
            "ifeval",
            "leaderboard_ifeval",
            "gsm8k",
            "boolq",
            "arc_challenge",
            "piqa",
        ),
    ),
    (
        "Preserve CPT Korean knowledge gains",
        (
            "global_mmlu_full_ko_medical_genetics",
            "global_mmlu_full_ko_nutrition",
            "global_mmlu_full_ko_philosophy",
            "global_mmlu_full_ko_miscellaneous",
        ),
    ),
    (
        "Recover CPT multiple-choice and exact-answer regressions",
        (
            "global_mmlu_full_ko_professional_medicine",
            "global_mmlu_full_ko_high_school_statistics",
            "global_mmlu_full_ko_astronomy",
            "global_mmlu_full_ko_high_school_computer_science",
            "global_mmlu_full_ko_jurisprudence",
            "kmmlu_direct_hard",
            "kmmlu_direct_hard_stem",
            "mmlu_prox_lite_ko",
            "mmlu_pro_law",
            "mmlu_pro_economics",
        ),
    ),
)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def iter_results(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("results_*.json")
        if not path.name.startswith("samples_")
    )


def load_reference_scores(path: Path) -> dict[str, dict[str, tuple[str, float]]]:
    data = load_json(path)
    if not data:
        return {}
    refs: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    for task, row in data.get("tasks", {}).items():
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric", ""))
        for model in ("Base", "CPT"):
            value = row.get(model)
            if isinstance(value, (int, float)):
                refs[task][model] = (metric, float(value))
    return refs


def model_alias(data: dict[str, Any], path: Path) -> str:
    name = data.get("model_name") or ""
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    path_text = str(path)
    if "KO-SFT-stage2-4k-diverse-kotsqa" in path_text:
        return "KO-SFT"
    if "KO-Agentic-SFT" in path_text:
        return "Agentic-SFT"
    if "KO-CPT-FULL" in path_text:
        return "CPT"
    if "LiquidAI__LFM2.5-8B-A1B" in path_text:
        return "Base"
    return name or path.parent.name


def pick_metric(task: str, metrics: dict[str, Any]) -> tuple[str, float] | None:
    for key in METRIC_PRIORITY:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not key.endswith("_stderr"):
            return key, float(value)
    return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.4f}"


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            "usage: summarize_linkedin_benchmark_results.py <eval_root> [reference_json]",
            file=sys.stderr,
        )
        return 2

    root = Path(sys.argv[1])
    ref_path = Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_REFERENCE
    scores: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    for task, row in load_reference_scores(ref_path).items():
        scores[task].update(row)
    samples: dict[str, Any] = {}

    for path in iter_results(root):
        data = load_json(path)
        if not data or "results" not in data:
            continue
        model = model_alias(data, path)
        n_samples = data.get("n-samples", {})
        for task, metrics in data["results"].items():
            if not isinstance(metrics, dict):
                continue
            picked = pick_metric(task, metrics)
            if picked is None:
                continue
            metric, value = picked
            scores[task][model] = (metric, value)
            if isinstance(n_samples, dict) and task in n_samples:
                samples[task] = n_samples[task]

    print("# Selected Full Benchmark Summary")
    print()
    print(f"Root: `{root}`")
    print(f"Reference: `{ref_path}`")
    print()
    print("This table is intended for public reporting after the selected SFT full runs finish.")
    print("Base/CPT columns are loaded from the CPT model-card reference where available; KO-SFT is freshly evaluated.")
    print()

    for section, tasks in SECTION_ORDER:
        print(f"## {section}")
        print()
        print("| task | metric | n | Base | CPT | KO-SFT | SFT-Base | SFT-CPT |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        for task in tasks:
            row = scores.get(task, {})
            metric = next((row[m][0] for m in ("KO-SFT", "CPT", "Base") if m in row), "")
            base = row.get("Base", ("", None))[1]
            cpt = row.get("CPT", ("", None))[1]
            sft = row.get("KO-SFT", ("", None))[1]
            n_value = samples.get(task, "")
            if isinstance(n_value, dict):
                n_value = n_value.get("effective", n_value.get("original", ""))
            print(
                f"| {TASK_LABELS.get(task, task)} | `{metric}` | {n_value} | "
                f"{fmt(base)} | {fmt(cpt)} | {fmt(sft)} | {delta(sft, base)} | {delta(sft, cpt)} |"
            )
        print()

    missing = [
        task
        for _, tasks in SECTION_ORDER
        for task in tasks
        if "KO-SFT" not in scores.get(task, {})
    ]
    if missing:
        print("## Missing / Pending")
        print()
        for task in missing:
            print(f"- `{task}`")
        print()

    print("## Reporting Notes")
    print()
    print("- Use this only after the selected KO-SFT full runs have completed; do not mix with LIMIT=50 gate runs.")
    print("- Base/CPT values come from the CPT card reference JSON and are not re-run in this fast path.")
    print("- BFCLv3/v4, Tau2, IFBench, Multi-IF, AA-Omniscience, MATH500, and AIME25 are official-card axes but are separated into slower supplement runs here.")
    print("- The Korean SFT success criterion is preserving CPT gains while recovering MCQA/exact-answer reliability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
