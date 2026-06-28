#!/usr/bin/env python3
"""Convert LFM-style text JSONL rows into HRM instruction/response JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\n(?P<response>.*?)(?:<\|im_end\|>|$)", re.DOTALL)


def split_text(text: str) -> tuple[str, str] | None:
    matches = list(ASSISTANT_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    response = last.group("response").strip()
    instruction = text[: last.start()].strip()
    if not instruction or not response:
        return None
    return instruction, response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--condition", default="direct")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--min-response-chars", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "input": str(inp),
        "output": str(out),
        "source_id": args.source_id,
        "read": 0,
        "written": 0,
        "dropped_parse": 0,
        "dropped_short_response": 0,
        "duplicate": 0,
    }
    seen: set[str] = set()
    with inp.open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            stats["read"] += 1
            if args.max_rows and stats["read"] > args.max_rows:
                break
            obj = json.loads(line)
            text = str(obj.get("text") or "")
            pair = split_text(text)
            if pair is None:
                stats["dropped_parse"] += 1
                continue
            instruction, response = pair
            if len(response) < args.min_response_chars:
                stats["dropped_short_response"] += 1
                continue
            digest = hashlib.sha256((instruction + "\n" + response).encode("utf-8", errors="ignore")).hexdigest()
            if args.dedupe and digest in seen:
                stats["duplicate"] += 1
                continue
            seen.add(digest)
            row = {
                "instruction": instruction,
                "response": response,
                "condition": args.condition,
                "source": args.source_id,
                "category": obj.get("category"),
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written"] += 1
    out.with_suffix(out.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
