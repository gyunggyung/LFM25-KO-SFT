#!/usr/bin/env python3
"""Build CPU-only raw JSONL for a small KO-CPT repair SFT run.

This script does not train. It creates instruction/response JSONL rows focused
on exact Korean MCQA answers, compact JSON answers, and short grounded answers.
The output can be tokenized later with prepare_lfm_chat_sft_data.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


MESSAGE_RE = re.compile(r"<\|im_start\|>([a-zA-Z_]+)\n(.*?)<\|im_end\|>", re.DOTALL)
END_TOKEN = "<|im_end|>"
CHOICE_RE = re.compile(r'"answer_choice"\s*:\s*"([^"]+)"')


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_lfm_messages(text: str) -> list[tuple[str, str]]:
    return [(role, content.strip()) for role, content in MESSAGE_RE.findall(text)]


def get_last(messages: list[tuple[str, str]], role: str) -> str:
    for msg_role, content in reversed(messages):
        if msg_role == role:
            return content
    return ""


def parse_jsonish_answer(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.endswith(END_TOKEN):
        cleaned = cleaned[: -len(END_TOKEN)].strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = CHOICE_RE.search(cleaned)
        return {"answer_choice": match.group(1)} if match else {}


def normalize_choice(value: Any) -> str:
    choice = str(value or "").strip()
    choice = choice.replace("번", "").strip()
    return choice


def legal_reason(answer: dict[str, Any]) -> str:
    basis = answer.get("legal_basis")
    if isinstance(basis, list) and basis:
        first = basis[0]
        if isinstance(first, dict):
            reason = str(first.get("reason") or "").strip()
            if reason:
                return reason
    notes = str(answer.get("notes") or "").strip()
    if notes:
        return notes[:240]
    return "제시된 근거 후보와 문제 문언을 대조해 정답을 선택한다."


def clean_user_for_answer_only(user: str) -> str:
    user = user.replace("위 후보 근거만 사용해 정답을 판단하고 JSON 하나만 출력하라.", "")
    user = user.strip()
    return (
        "다음 한국어 다지선다 문제를 풀어라. "
        "정답 번호만 출력하고 해설은 쓰지 마라.\n\n"
        f"{user}\n\n정답 번호만 출력:"
    )


def clean_user_for_json(user: str) -> str:
    user = user.strip()
    return (
        "다음 한국어 다지선다 문제를 풀어라. "
        "반드시 JSON 하나만 출력하라. 스키마: {\"answer_choice\":\"정답 번호\"}\n\n"
        f"{user}"
    )


def clean_user_for_rationale(user: str) -> str:
    user = user.replace("위 후보 근거만 사용해 정답을 판단하고 JSON 하나만 출력하라.", "")
    user = user.strip()
    return (
        "다음 한국어 다지선다 문제를 풀어라. "
        "짧은 근거를 한 문장으로 쓴 뒤 마지막 줄에 `정답: 번호` 형식으로 답하라.\n\n"
        f"{user}"
    )


def add_record(
    records: list[dict[str, Any]],
    seen: set[str],
    instruction: str,
    response: str,
    source: str,
    category: str,
    metadata: dict[str, Any],
) -> None:
    instruction = instruction.strip()
    response = response.strip()
    if not instruction or not response:
        return
    digest = hashlib.sha256((instruction + "\n" + response).encode("utf-8", errors="ignore")).hexdigest()
    if digest in seen:
        return
    seen.add(digest)
    records.append(
        {
            "instruction": instruction,
            "response": response,
            "source": source,
            "category": category,
            "metadata": metadata,
        }
    )


def build_legal_records(path: Path, max_rows: int, seen: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rows = 0
    for row in iter_jsonl(path):
        rows += 1
        if max_rows and rows > max_rows:
            break
        messages = parse_lfm_messages(str(row.get("text") or ""))
        user = get_last(messages, "user")
        assistant = get_last(messages, "assistant")
        answer = parse_jsonish_answer(assistant)
        choice = normalize_choice(answer.get("answer_choice"))
        if not user or not choice:
            continue
        source = str(row.get("source") or "current_law_bar_json_answer_sft_20260621")
        meta = {"row": rows, "variant_family": "legal_bar", "original_category": row.get("category")}

        add_record(
            records,
            seen,
            clean_user_for_answer_only(user),
            choice,
            source,
            "korean_mcqa_answer_only",
            meta | {"variant": "answer_only"},
        )
        add_record(
            records,
            seen,
            clean_user_for_json(user),
            json.dumps({"answer_choice": choice}, ensure_ascii=False),
            source,
            "korean_mcqa_json_compact",
            meta | {"variant": "json_compact"},
        )
        add_record(
            records,
            seen,
            clean_user_for_rationale(user),
            f"근거: {legal_reason(answer)}\n정답: {choice}",
            source,
            "korean_mcqa_short_rationale",
            meta | {"variant": "short_rationale"},
        )
    return records


def first_answer_line(text: str) -> str:
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    return lines[0] if lines else ""


def build_kotsqa_records(path: Path, max_rows: int, seen: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rows = 0
    for row in iter_jsonl(path):
        rows += 1
        if max_rows and rows > max_rows:
            break
        instruction = str(row.get("instruction") or "").strip()
        response = first_answer_line(str(row.get("response") or ""))
        if not instruction or not response:
            continue
        add_record(
            records,
            seen,
            instruction + "\n\n정답만 짧게 출력:",
            response,
            str(row.get("source") or "etri_lirs_kotsqa_v2_train"),
            "korean_evidence_qa_short_answer",
            {
                "row": rows,
                "dataset": "etri-lirs/KoTSQA-v.2.0",
                "split": "train",
                "source_id": row.get("id"),
            },
        )
    return records


def write_outputs(records: list[dict[str, Any]], out_path: Path, seed: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(records)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_category: dict[str, int] = {}
    for record in records:
        by_category[record["category"]] = by_category.get(record["category"], 0) + 1
    stats = {
        "output": str(out_path),
        "rows": len(records),
        "by_category": by_category,
        "seed": seed,
    }
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legal-bar-jsonl", action="append", default=[])
    parser.add_argument("--kotsqa-jsonl", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-legal-rows", type=int, default=0)
    parser.add_argument("--max-kotsqa-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=60630)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in args.legal_bar_jsonl:
        records.extend(build_legal_records(Path(item), args.max_legal_rows, seen))
    for item in args.kotsqa_jsonl:
        records.extend(build_kotsqa_records(Path(item), args.max_kotsqa_rows, seen))
    if not records:
        raise SystemExit("no repair records built")
    write_outputs(records, Path(args.output), args.seed)


if __name__ == "__main__":
    main()
