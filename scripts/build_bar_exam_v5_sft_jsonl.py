#!/usr/bin/env python3
"""Build bar-exam v5 context-grounded SFT rows.

This is a CPU-only data builder. It does not train a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SUBJECT_EN_TO_KO = {
    "civil_law": "민사법",
    "criminal_law": "형사법",
    "public_law": "공법",
}
SUBJECT_KO_TO_EN = {v: k for k, v in SUBJECT_EN_TO_KO.items()}
CHOICE_NORMALIZE = {
    "①": "1",
    "②": "2",
    "③": "3",
    "④": "4",
    "⑤": "5",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
}
CHOICE_SYMBOLS = ["①", "②", "③", "④", "⑤"]
ANSWER_RE = re.compile(r"최종\s*답\s*:\s*([1-5])\s*번")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text).replace("\r\n", "\n").strip())


def lfm_text(messages: list[dict[str, str]]) -> str:
    chunks = ["<|startoftext|>"]
    for msg in messages:
        chunks.append(f"<|im_start|>{msg['role']}\n{msg['content'].strip()}<|im_end|>\n")
    return "".join(chunks)


def get_first_key(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name])
    return default


def parse_choices(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return []


def normalize_answer(answer: str) -> str | None:
    answer = str(answer).strip()
    return CHOICE_NORMALIZE.get(answer)


def format_choices_symbol(choices: list[str]) -> str:
    lines = []
    for idx, choice in enumerate(choices[:5]):
        text = re.sub(r"^[①②③④⑤]\s*", "", choice.strip())
        lines.append(f"{CHOICE_SYMBOLS[idx]} {text}")
    return "\n".join(lines)


def format_choices_numeric(choices: list[str]) -> str:
    lines = []
    for idx, choice in enumerate(choices[:5], start=1):
        text = re.sub(r"^[①②③④⑤]\s*", "", choice.strip())
        lines.append(f"{idx}. {text}")
    return "\n".join(lines)


def build_mcqa_prompt(stem: str, question_text: str, choices: list[str], style: str) -> str:
    if choices:
        choice_block = format_choices_numeric(choices) if style == "numeric" else format_choices_symbol(choices)
        base_question = stem.strip() or question_text.strip()
        question_text = f"{base_question}\n\n{choice_block}"
    return clean_text(
        "다음 변호사시험 선택형 문제를 풀어라. 마지막 줄에는 정답 번호만 `정답: N` 형식으로 써라.\n\n"
        f"{question_text}"
    )


def build_mcqa_rows(questions_csv: Path, include_round15_labels: bool) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = Counter()
    by_round = defaultdict(Counter)
    with questions_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["raw_questions"] += 1
            round_no = int(get_first_key(row, "round", default="0") or 0)
            subject = get_first_key(row, "subject")
            question_no = get_first_key(row, "question_no")
            raw_answer = get_first_key(row, "answer")
            answer = normalize_answer(raw_answer)
            by_round[str(round_no)]["raw"] += 1
            if answer is None:
                stats["mcqa_skipped_unsafe_answer"] += 1
                by_round[str(round_no)]["skipped_unsafe_answer"] += 1
                continue
            if round_no == 15 and not include_round15_labels:
                stats["mcqa_skipped_round15_holdout"] += 1
                by_round[str(round_no)]["skipped_holdout"] += 1
                continue
            stem = get_first_key(row, "stem")
            question_text = get_first_key(row, "question_text") or stem
            choices = parse_choices(get_first_key(row, "choices_json"))
            if not question_text.strip():
                stats["mcqa_skipped_no_question"] += 1
                continue

            for style in ("symbol", "numeric"):
                instruction = build_mcqa_prompt(stem, question_text, choices, style)
                rows.append(
                    {
                        "source": "bar_exam_processed_multiple_choice",
                        "category": f"mcqa_safe_answer_{style}",
                        "instruction": instruction,
                        "response": f"정답: {answer}",
                        "metadata": {
                            "round": round_no,
                            "subject": subject,
                            "question_no": question_no,
                            "raw_answer": raw_answer,
                            "answer": answer,
                            "answer_safe": True,
                            "round15_answer_label": round_no == 15,
                        },
                    }
                )
                stats[f"mcqa_rows_{style}"] += 1
            by_round[str(round_no)]["kept_safe_questions"] += 1
    return rows, {"totals": dict(stats), "by_round": {k: dict(v) for k, v in sorted(by_round.items(), key=lambda x: int(x[0]))}}


def messages_to_lfm_row(row: dict, source: str, category: str) -> dict | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    converted = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            return None
        converted.append({"role": role, "content": content})
    return {
        "source": source,
        "category": category,
        "text": lfm_text(converted),
        "metadata": {
            "id": row.get("id", ""),
            "subject": row.get("subject", ""),
            "law_title": row.get("law_title", ""),
            "article": row.get("article", ""),
        },
    }


def build_current_law_rows(paths: list[Path]) -> tuple[list[dict], dict]:
    rows = []
    stats = Counter()
    for path in paths:
        source = path.parent.parent.parent.name
        category = "current_law_bar_hard" if "hard" in str(path) else "current_law_bar_simple"
        for raw in iter_jsonl(path):
            stats[f"raw_{category}"] += 1
            converted = messages_to_lfm_row(raw, source, category)
            if converted:
                rows.append(converted)
                stats[f"rows_{category}"] += 1
            else:
                stats[f"skipped_{category}"] += 1
    return rows, dict(stats)


def subject_file_stem(subject_en: str, question_no: int) -> str:
    return f"{subject_en}_{question_no:02d}"


def context_path_for(v5_dir: Path, subject_en: str, question_no: int) -> Path:
    subject_ko = SUBJECT_EN_TO_KO[subject_en]
    return v5_dir / f"q{question_no:03d}_{subject_ko}.md"


def split_path_for(split_dir: Path, subject_en: str, question_no: int) -> Path:
    return split_dir / subject_en / f"{subject_en}_{question_no:02d}.json"


def solution_paths(solved_dir: Path) -> Iterable[Path]:
    yield from sorted(solved_dir.glob("*/*_solution.md"))


def extract_answer_from_solution(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    return match.group(1) if match else None


def render_problem_from_split(problem: dict) -> str:
    question = str(problem.get("question", "")).strip()
    passage = problem.get("passage")
    material = problem.get("material")
    underlines = problem.get("underlines")
    propositions = problem.get("propositions")
    choices = problem.get("choices")

    blocks = []
    if passage:
        blocks.append(f"[지문]\n{passage}")
    if material:
        blocks.append(f"[자료]\n{material}")
    if underlines:
        blocks.append(f"[밑줄]\n{json.dumps(underlines, ensure_ascii=False, indent=2)}")
    blocks.append(f"[문제]\n{question}")
    if isinstance(propositions, dict):
        blocks.append("[보기]\n" + "\n".join(f"{k}. {v}" for k, v in propositions.items()))
    if isinstance(choices, dict):
        blocks.append("[선택지]\n" + "\n".join(f"{k} {v}" for k, v in choices.items()))
    return clean_text("\n\n".join(blocks))


def build_v5_solver_rows(split_dir: Path, v5_context_dir: Path, solved_dir: Path) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = Counter()
    by_subject = defaultdict(Counter)
    for sol_path in solution_paths(solved_dir):
        stats["solution_files"] += 1
        subject_en = sol_path.parent.name
        question_no = int(re.search(r"_(\d+)_solution\.md$", sol_path.name).group(1))
        meta_path = sol_path.with_name(sol_path.name.replace("_solution.md", ".json"))
        problem_path = split_path_for(split_dir, subject_en, question_no)
        context_path = context_path_for(v5_context_dir, subject_en, question_no)
        by_subject[subject_en]["solution_files"] += 1
        if not meta_path.exists() or not problem_path.exists() or not context_path.exists():
            stats["skipped_missing_pair"] += 1
            by_subject[subject_en]["skipped_missing_pair"] += 1
            continue
        solution = clean_text(sol_path.read_text(encoding="utf-8"))
        answer = extract_answer_from_solution(solution)
        meta = read_json(meta_path)
        if not answer:
            answer = normalize_answer(str(meta.get("answer", "")))
        if answer not in {"1", "2", "3", "4", "5"}:
            stats["skipped_no_safe_answer"] += 1
            by_subject[subject_en]["skipped_no_safe_answer"] += 1
            continue
        problem = read_json(problem_path)
        context = clean_text(context_path.read_text(encoding="utf-8"))
        prompt = clean_text(
            "다음은 변호사시험 선택형 문제와 v5 근거 패킷이다. 근거 패킷 안의 법령ㆍ판례ㆍ수동 검증 포인트만 우선하여 "
            "각 지문 또는 선택지를 판단하고, 마지막 줄에 `최종 답: N번.` 형식으로 결론을 내라. "
            "근거가 충돌하면 수동 검증 보강 포인트를 우선하고, 답안 번호는 1부터 5 사이의 숫자로만 확정하라.\n\n"
            f"{render_problem_from_split(problem)}\n\n[근거 패킷]\n{context}"
        )
        rows.append(
            {
                "source": "bar_exam_15th_solved_v5",
                "category": "v5_context_grounded_full_solution",
                "instruction": prompt,
                "response": solution,
                "metadata": {
                    "round": 15,
                    "subject": SUBJECT_EN_TO_KO[subject_en],
                    "subject_en": subject_en,
                    "question_no": question_no,
                    "answer": answer,
                    "answer_from": str(meta_path),
                    "problem_path": str(problem_path),
                    "context_path": str(context_path),
                    "solution_path": str(sol_path),
                    "uses_15th_answer_label": True,
                    "requires_context": True,
                },
            }
        )
        stats["rows_v5_full_solution"] += 1
        by_subject[subject_en]["rows_v5_full_solution"] += 1

        answer_only_solution = f"정답: {answer}\n\n근거 요약:\n" + "\n".join(
            line for line in solution.splitlines() if "최종 답" in line or "| " in line
        )[:1600]
        rows.append(
            {
                "source": "bar_exam_15th_solved_v5",
                "category": "v5_context_grounded_answer_compact",
                "instruction": prompt
                + "\n\n출력은 짧게 작성하라. 선택지 판단 근거를 요약하고 마지막 줄에 `정답: N`만 한 번 더 써라.",
                "response": answer_only_solution + f"\n\n정답: {answer}",
                "metadata": {
                    "round": 15,
                    "subject": SUBJECT_EN_TO_KO[subject_en],
                    "subject_en": subject_en,
                    "question_no": question_no,
                    "answer": answer,
                    "uses_15th_answer_label": True,
                    "requires_context": True,
                },
            }
        )
        stats["rows_v5_answer_compact"] += 1
        by_subject[subject_en]["rows_v5_answer_compact"] += 1
    return rows, {"totals": dict(stats), "by_subject": {k: dict(v) for k, v in sorted(by_subject.items())}}


def build_v5_answer_free_rows(v5_context_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    stats = Counter()
    for path in sorted(v5_context_dir.glob("q*.md")):
        stats["context_files"] += 1
        text = clean_text(path.read_text(encoding="utf-8"))
        m = re.match(r"q(\d{3})_(.+)\.md$", path.name)
        if not m:
            continue
        qno = int(m.group(1))
        subject = m.group(2)
        prompt = clean_text(
            "다음 v5 근거 패킷을 읽고, 최종 정답 번호를 내지 말고 문제 풀이 절차만 정리하라. "
            "반드시 어떤 법령ㆍ판례ㆍ수동 검증 포인트를 먼저 볼지, 선택지를 어떻게 대조할지 설명하라.\n\n"
            f"{text}"
        )
        response = clean_text(
            "풀이 절차:\n"
            "1. 문제 방향이 `옳은 것`, `옳지 않은 것`, `모두 고른 것` 중 무엇인지 먼저 확정한다.\n"
            "2. v5 수동 검증 보강 포인트를 자동 검색 후보보다 우선한다.\n"
            "3. 각 지문 또는 선택지별로 O/X를 표시하고, 근거가 되는 법령ㆍ판례 문장을 짧게 붙인다.\n"
            "4. 조합형 문항이면 목표 지문 집합과 선택지를 대조한다.\n"
            "5. 마지막 검산에서는 선택지 번호와 지문 집합이 서로 일치하는지만 재확인한다.\n"
            "이 샘플은 절차 학습용이므로 최종 정답 번호는 출력하지 않는다."
        )
        rows.append(
            {
                "source": "bar_exam_round15_rag_contexts_v5",
                "category": "v5_answer_free_procedure",
                "instruction": prompt,
                "response": response,
                "metadata": {
                    "round": 15,
                    "subject": subject,
                    "question_no": qno,
                    "answer_free": True,
                    "context_path": str(path),
                },
            }
        )
        stats["rows"] += 1
    return rows, dict(stats)


def build_search_action_rows(path: Path) -> tuple[list[dict], dict]:
    rows = []
    stats = Counter()
    if not path.exists():
        return rows, {"missing": str(path)}
    for raw in iter_jsonl(path):
        stats["raw"] += 1
        messages = raw.get("messages")
        if not isinstance(messages, list):
            stats["skipped_no_messages"] += 1
            continue
        prefix: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") in {"system", "user"}:
                prefix.append({"role": msg["role"], "content": str(msg.get("content", "")).strip()})
            elif msg.get("role") == "assistant":
                content = str(msg.get("content", "")).strip()
                if not content:
                    break
                prefix.append({"role": "assistant", "content": content})
                rows.append(
                    {
                        "source": "bar_exam_round150_rag_contexts_v2",
                        "category": "legal_search_first_action",
                        "text": lfm_text(prefix),
                        "metadata": {"answer_free": True, "id": raw.get("id", "")},
                    }
                )
                stats["rows"] += 1
                break
    return rows, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar-exam-root", required=True)
    parser.add_argument("--current-law-jsonl", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument(
        "--mode",
        choices=["holdout_clean", "context_solver", "product_tuned"],
        default="context_solver",
        help="holdout_clean excludes 15th answer labels; context_solver includes v5 solved context rows; product_tuned also includes safe 15th MCQA labels.",
    )
    args = parser.parse_args()

    root = Path(args.bar_exam_root)
    rows: list[dict] = []
    stats: dict[str, object] = {"mode": args.mode, "sources": {}}

    include_round15_mcqa = args.mode == "product_tuned"
    mcqa_rows, mcqa_stats = build_mcqa_rows(root / "processed_multiple_choice" / "questions.csv", include_round15_mcqa)
    rows.extend(mcqa_rows)
    stats["sources"]["processed_multiple_choice"] = mcqa_stats

    current_rows, current_stats = build_current_law_rows([Path(p) for p in args.current_law_jsonl])
    rows.extend(current_rows)
    stats["sources"]["current_law"] = current_stats

    free_rows, free_stats = build_v5_answer_free_rows(root / "round15_rag_contexts_v5_20260629")
    rows.extend(free_rows)
    stats["sources"]["round15_rag_contexts_v5_answer_free"] = free_stats

    search_rows, search_stats = build_search_action_rows(root / "round150_rag_contexts_v2_20260621" / "sft" / "bar_exam_fts_agent_sft.jsonl")
    rows.extend(search_rows)
    stats["sources"]["round150_search_actions"] = search_stats

    if args.mode in {"context_solver", "product_tuned"}:
        solver_rows, solver_stats = build_v5_solver_rows(
            root / "15th_split",
            root / "round15_rag_contexts_v5_20260629",
            root / "15th_solved_v5",
        )
        rows.extend(solver_rows)
        stats["sources"]["15th_solved_v5"] = solver_stats

    stats["total_rows"] = write_jsonl(Path(args.output), rows)
    stats["category_counts"] = dict(Counter(row.get("category", "") for row in rows))
    stats["source_counts"] = dict(Counter(row.get("source", "") for row in rows))
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
