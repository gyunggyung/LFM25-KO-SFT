#!/usr/bin/env python3
"""Convert Fable-style Korean message JSONL into LFM chat JSONL.

The output rows use a single `text` field with LFM ChatML markers. The existing
`prepare_lfm_chat_sft_data.py` script then builds response-only labels from the
last assistant turn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


START = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
OVERCLAIM_RE = re.compile(
    r"(mythos급|mythos|초당\s*\d|ops/sec|벤치마크.*\d+x|formal verification|kani|miri)",
    re.IGNORECASE,
)


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                if line.strip():
                    yield path, idx, json.loads(line)


def clean_content(role: str, content: Any, strip_think: bool) -> str:
    text = str(content or "").strip()
    if strip_think and role == "assistant":
        text = THINK_BLOCK_RE.sub("", text).strip()
    return text


def normalize_role(role: Any) -> str:
    value = str(role or "user").strip().lower()
    if value in {"system", "user", "assistant", "tool"}:
        return value
    return "user"


def messages_to_lfm_text(messages: list[dict[str, Any]], strip_think: bool) -> tuple[str, int] | None:
    parts: list[str] = [START]
    assistant_count = 0

    for msg in messages:
        role = normalize_role(msg.get("role"))
        content = clean_content(role, msg.get("content"), strip_think)
        if not content:
            continue
        if role == "assistant":
            assistant_count += 1
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")

    if assistant_count == 0:
        return None
    return "".join(parts), assistant_count


def source_name(path: Path, row: dict[str, Any]) -> str:
    for key in ("dataset", "source"):
        value = row.get(key)
        if value:
            return str(value)
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strip-think", action="store_true", default=True)
    parser.add_argument("--keep-think", action="store_false", dest="strip_think")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--drop-overclaim", action="store_true", default=True)
    parser.add_argument("--keep-overclaim", action="store_false", dest="drop_overclaim")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    inputs = [Path(x) for x in args.input]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stats = {
        "rows": 0,
        "kept": 0,
        "dropped_parse": 0,
        "dropped_overclaim": 0,
        "dedupe_dropped": 0,
        "assistant_turns": 0,
        "inputs": [str(x) for x in inputs],
        "strip_think": args.strip_think,
        "drop_overclaim": args.drop_overclaim,
    }

    with out.open("w", encoding="utf-8") as handle:
        for path, row_idx, row in iter_jsonl(inputs):
            stats["rows"] += 1
            if args.max_rows and stats["rows"] > args.max_rows:
                break

            messages = row.get("messages")
            if not isinstance(messages, list):
                stats["dropped_parse"] += 1
                continue

            if args.drop_overclaim:
                joined = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
                if OVERCLAIM_RE.search(joined):
                    stats["dropped_overclaim"] += 1
                    continue

            converted = messages_to_lfm_text(messages, args.strip_think)
            if converted is None:
                stats["dropped_parse"] += 1
                continue
            text, assistant_turns = converted

            digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            if args.dedupe and digest in seen:
                stats["dedupe_dropped"] += 1
                continue
            seen.add(digest)

            stats["kept"] += 1
            stats["assistant_turns"] += assistant_turns
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "source": source_name(path, row),
                        "src_path": str(path),
                        "src_row": row.get("src_row", row_idx),
                        "conversion": "fable_agentic_to_lfm_chat",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
