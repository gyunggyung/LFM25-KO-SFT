#!/usr/bin/env python3
"""Convert KoTSQA-v2.0 rows into LFM chat SFT JSONL.

The output uses the plain instruction/response shape accepted by
prepare_lfm_chat_sft_data.py. Keep the official test split out of training by
default so we can reuse it for later Korean QA evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def normalize_answer(answers: object) -> str:
    if isinstance(answers, list):
        vals = [str(x).strip() for x in answers if str(x).strip()]
    elif answers is None:
        vals = []
    else:
        vals = [str(answers).strip()]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    # Preserve alternate labels without making the assistant verbose.
    return vals[0] + "\n\n허용 가능한 다른 정답: " + "; ".join(vals[1:])


def trim_passage(text: object, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars > 0 and len(value) > max_chars:
        return value[: max_chars - 20].rstrip() + " ... [중략]"
    return value


def build_instruction(row: dict, max_passages: int, max_passage_chars: int) -> str:
    passages = row.get("passages") or []
    if not isinstance(passages, list):
        passages = [passages]
    clipped = []
    for i, passage in enumerate(passages[:max_passages], 1):
        text = trim_passage(passage, max_passage_chars)
        if text:
            clipped.append(f"[문서 {i}]\n{text}")
    context = "\n\n".join(clipped)
    fact_type = row.get("fact_type", "")
    q_type = row.get("q_type", "")
    return (
        "아래 문서들을 근거로 한국어 질문에 답하세요. "
        "정답만 짧고 정확하게 말하되, 질문 전제가 틀렸다면 문서 근거에 맞게 바로잡아 답하세요.\n\n"
        f"질문 유형: {q_type}\n"
        f"사실 유형: {fact_type}\n\n"
        f"{context}\n\n"
        f"질문: {str(row.get('question', '')).strip()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="etri-lirs/KoTSQA-v.2.0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-passages", type=int, default=4)
    parser.add_argument("--max-passage-chars", type=int, default=1800)
    parser.add_argument("--source-id", default="etri_lirs_kotsqa_v2_train")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    rows = 0
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            rows += 1
            if args.max_rows and rows > args.max_rows:
                break
            question = str(row.get("question", "")).strip()
            answer = normalize_answer(row.get("answers"))
            if not question or not answer:
                continue
            record = {
                "id": str(row.get("id", f"kotsqa_{rows}")),
                "source": args.source_id,
                "instruction": build_instruction(row, args.max_passages, args.max_passage_chars),
                "response": answer,
                "metadata": {
                    "dataset": args.dataset,
                    "split": args.split,
                    "fact_type": row.get("fact_type"),
                    "q_type": row.get("q_type"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "rows": rows,
        "kept": kept,
        "output": str(out_path),
        "max_passages": args.max_passages,
        "max_passage_chars": args.max_passage_chars,
    }
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
