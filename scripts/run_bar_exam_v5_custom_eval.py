#!/usr/bin/env python3
"""Run a custom 15th bar-exam v5 grounded MCQA evaluation.

The evaluator talks to an OpenAI-compatible endpoint, so it can be used with
vLLM, llama.cpp server, or any compatible backend.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

from openai import OpenAI


SUBJECT_EN_TO_KO = {
    "civil_law": "민사법",
    "criminal_law": "형사법",
    "public_law": "공법",
}
ANSWER_RE = re.compile(
    r"(?:정답|최종\s*답|답)\s*[:：]?\s*([1-5])\s*(?:번)?|([①②③④⑤])"
)
CHOICE_TO_NUM = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text).replace("\r\n", "\n").strip())


def render_problem(problem: dict) -> str:
    blocks: list[str] = []
    if problem.get("passage"):
        blocks.append(f"[지문]\n{problem['passage']}")
    if problem.get("material"):
        blocks.append(f"[자료]\n{problem['material']}")
    if problem.get("underlines"):
        blocks.append("[밑줄]\n" + json.dumps(problem["underlines"], ensure_ascii=False, indent=2))
    blocks.append(f"[문제]\n{problem.get('question', '')}")
    propositions = problem.get("propositions")
    if isinstance(propositions, dict):
        blocks.append("[보기]\n" + "\n".join(f"{k}. {v}" for k, v in propositions.items()))
    choices = problem.get("choices")
    if isinstance(choices, dict):
        blocks.append("[선택지]\n" + "\n".join(f"{k} {v}" for k, v in choices.items()))
    return clean_text("\n\n".join(blocks))


def extract_context_focus(text: str, max_chars: int) -> str:
    """Keep the high-signal v5 section first and cap prompt length."""
    text = clean_text(text)
    markers = [
        "## v5 수동 검증 보강 포인트",
        "## 법령 근거",
        "## 판례 근거",
    ]
    start = text.find(markers[0])
    if start >= 0:
        focused = text[start:]
    else:
        focused = text

    if len(focused) <= max_chars:
        return focused

    chunks: list[str] = []
    for marker in markers:
        pos = text.find(marker)
        if pos < 0:
            continue
        next_positions = [text.find(m, pos + 1) for m in markers if text.find(m, pos + 1) > pos]
        end = min(next_positions) if next_positions else len(text)
        chunks.append(text[pos:end].strip())
    focused = "\n\n".join(chunks) if chunks else text
    return focused[:max_chars].rstrip()


def answer_from_text(text: str) -> str | None:
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    value = match.group(1) or CHOICE_TO_NUM.get(match.group(2), "")
    return value if value in {"1", "2", "3", "4", "5"} else None


def iter_cases(root: Path, context_root: Path, solved_root: Path, limit: int | None) -> Iterable[dict]:
    count = 0
    for subject_en in ("civil_law", "criminal_law", "public_law"):
        subject_ko = SUBJECT_EN_TO_KO[subject_en]
        for meta_path in sorted((solved_root / subject_en).glob("*.json")):
            m = re.search(r"_(\d+)\.json$", meta_path.name)
            if not m:
                continue
            qno = int(m.group(1))
            problem_path = root / "15th_split" / subject_en / f"{subject_en}_{qno:02d}.json"
            context_path = context_root / f"q{qno:03d}_{subject_ko}.md"
            if not problem_path.exists() or not context_path.exists():
                continue
            meta = read_json(meta_path)
            answer = str(meta.get("answer", "")).strip()
            if answer not in {"1", "2", "3", "4", "5"}:
                continue
            yield {
                "id": f"{subject_en}_{qno:02d}",
                "subject_en": subject_en,
                "subject": subject_ko,
                "question_no": qno,
                "answer": answer,
                "problem_path": str(problem_path),
                "context_path": str(context_path),
                "problem": read_json(problem_path),
                "context": context_path.read_text(encoding="utf-8"),
            }
            count += 1
            if limit and count >= limit:
                return


def build_prompt(case: dict, max_context_chars: int) -> str:
    context = extract_context_focus(case["context"], max_context_chars)
    return clean_text(
        "다음은 변호사시험 선택형 문제와 v5 근거 패킷이다. "
        "근거 패킷의 법령, 판례, 수동 검증 보강 포인트만 우선하여 판단하라. "
        "마지막 줄은 반드시 `정답: 1`, `정답: 2`, `정답: 3`, `정답: 4`, `정답: 5` 중 하나로만 써라. "
        "`정답: N`, `정답: A`, `정답: B`, `정답: C`, `정답: D`, `정답: E`처럼 문자를 쓰면 오답이다.\n\n"
        f"{render_problem(case['problem'])}\n\n"
        f"[v5 근거 패킷]\n{context}"
    )


def build_strict_prompt(case: dict, max_context_chars: int) -> str:
    context = extract_context_focus(case["context"], max_context_chars)
    return clean_text(
        "다음 변호사시험 선택형 문제를 풀어라. v5 근거 패킷의 수동 검증 보강 포인트를 최우선으로 적용한다. "
        "출력은 숫자 하나만 허용된다. 다른 글자, 쉼표, 설명, `정답:` 접두어를 쓰지 마라. "
        "반드시 1, 2, 3, 4, 5 중 정확히 하나만 출력하라.\n\n"
        f"{render_problem(case['problem'])}\n\n"
        f"[v5 근거 패킷]\n{context}\n\n"
        "정답 숫자 하나:"
    )


def call_model(client: OpenAI, model: str, prompt: str, temperature: float, max_tokens: int, timeout_sleep: float) -> str:
    if timeout_sleep > 0:
        time.sleep(timeout_sleep)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "너는 한국 변호사시험 선택형 문제를 근거에 맞게 푸는 법률 평가 모델이다."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bar-exam-root", default="/home/work/.projects/LLM-OS-Models/Terminal/Bar-exam-test")
    parser.add_argument("--context-root", default="/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft/data/bar_exam/round15_rag_contexts_v5_20260629")
    parser.add_argument("--solved-root", default="/home/work/.projects/LLM-OS-Models/Terminal/Bar-exam-test/15th_solved_v5")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--strict-one-token", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url.rstrip("/") + "/v1", api_key=args.api_key)

    rows = []
    correct = 0
    extracted = 0
    total = 0
    for case in iter_cases(Path(args.bar_exam_root), Path(args.context_root), Path(args.solved_root), args.limit):
        prompt = (
            build_strict_prompt(case, args.max_context_chars)
            if args.strict_one_token
            else build_prompt(case, args.max_context_chars)
        )
        output = call_model(client, args.model, prompt, args.temperature, args.max_tokens, args.sleep)
        pred = answer_from_text(output)
        total += 1
        if pred:
            extracted += 1
        if pred == case["answer"]:
            correct += 1
        rows.append(
            {
                "id": case["id"],
                "subject": case["subject"],
                "question_no": case["question_no"],
                "gold": case["answer"],
                "pred": pred,
                "correct": pred == case["answer"],
                "output": output,
                "prompt_chars": len(prompt),
            }
        )
        print(f"{args.label} {case['id']} pred={pred} gold={case['answer']} correct={pred == case['answer']}", flush=True)

    result_path = out_dir / f"{args.label}.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "label": args.label,
        "model": args.model,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "extracted": extracted,
        "extraction_rate": extracted / total if total else 0.0,
        "result_path": str(result_path),
    }
    summary_path = out_dir / f"{args.label}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
