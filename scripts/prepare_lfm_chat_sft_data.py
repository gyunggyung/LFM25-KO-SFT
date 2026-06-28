#!/usr/bin/env python3
"""Prepare response-only SFT arrays with the native LFM tokenizer.

The older HRM prepared SFT sets in this workspace were built with a 131072
vocabulary tokenizer. LFM2.5-8B-A1B uses a 125017-token tokenizer, so those
arrays cannot be fed into the LFM model directly. This script rebuilds the
same start/length layout from either LFM-chat `text` JSONL rows or plain
`instruction`/`response` JSONL rows using the target LFM tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from array import array
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer


ASSISTANT_RE = re.compile(r"<\|im_start\|>assistant\n", re.DOTALL)
END_TOKEN = "<|im_end|>"
START_OF_TEXT = "<|startoftext|>"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
DEFAULT_USER_PREFIX = "<|startoftext|><|im_start|>user\n"
DEFAULT_USER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield path, json.loads(line)


def split_lfm_text(text: str) -> tuple[str, str] | None:
    matches = list(ASSISTANT_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    prefix = text[: last.end()]
    after = text[last.end() :]
    end_idx = after.find(END_TOKEN)
    if end_idx >= 0:
        response = after[:end_idx].strip() + END_TOKEN
    else:
        response = after.strip()
    if not prefix.strip() or not response.replace(END_TOKEN, "").strip():
        return None
    return prefix, response


def split_instruction_response(row: dict) -> tuple[str, str] | None:
    instruction = row.get("instruction")
    response = row.get("response")
    if instruction is None or response is None:
        return None
    instruction = str(instruction).strip()
    response = str(response).strip()
    if not instruction or not response:
        return None

    if "<|im_start|>" in instruction:
        prefix = instruction.rstrip()
        if not prefix.endswith(ASSISTANT_PREFIX.rstrip()):
            prefix = prefix + "\n" + ASSISTANT_PREFIX
        else:
            prefix = prefix + "\n"
    else:
        prefix = DEFAULT_USER_PREFIX + instruction + DEFAULT_USER_SUFFIX

    if not response.endswith(END_TOKEN):
        response = response + END_TOKEN
    return prefix, response


def truncate_middle(ids: list[int], budget: int, head_tokens: int) -> list[int]:
    if len(ids) <= budget:
        return ids
    if budget <= 0:
        return []
    head = min(head_tokens, budget // 2)
    tail = budget - head
    return ids[:head] + ids[-tail:]


def collect_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(x for x in p.rglob("*.jsonl") if x.is_file()))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(item)
    if not paths:
        raise ValueError("no input JSONL files")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=60628)
    parser.add_argument("--strip-think-blocks", action="store_true")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--truncate-head-tokens", type=int, default=1024)
    parser.add_argument("--min-response-tokens", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--target-tokens", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--source-id", default="")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_paths(args.train)
    tok = Tokenizer.from_file(args.tokenizer)

    all_tokens = array("i")
    inst_start: list[int] = []
    inst_len: list[int] = []
    resp_start: list[int] = []
    resp_len: list[int] = []
    seen: set[str] = set()
    max_id = -1

    stats = {
        "input_files": [str(p) for p in paths],
        "source_id": args.source_id,
        "rows": 0,
        "kept_rows": 0,
        "dropped_parse": 0,
        "dropped_empty_response": 0,
        "dropped_response_too_long": 0,
        "dedupe_dropped": 0,
        "truncated_rows": 0,
        "max_original_len": 0,
        "max_written_len": 0,
        "target_tokens": args.target_tokens,
        "max_seq_length": args.max_seq_length,
        "tokenizer_vocab_size": tok.get_vocab_size(with_added_tokens=True),
    }

    for _, row in iter_jsonl(paths):
        stats["rows"] += 1
        if args.max_rows and stats["rows"] > args.max_rows:
            break

        pair = None
        text = row.get("text")
        if text is not None:
            pair = split_lfm_text(str(text))
        if pair is None:
            pair = split_instruction_response(row)
        if pair is None:
            stats["dropped_parse"] += 1
            continue

        prefix, response = pair
        if args.strip_think_blocks:
            response = THINK_BLOCK_RE.sub("", response).strip()
            if response and not response.endswith(END_TOKEN):
                response += END_TOKEN
        if not response.replace(END_TOKEN, "").strip():
            stats["dropped_empty_response"] += 1
            continue

        digest = hashlib.sha256((prefix + "\n" + response).encode("utf-8", errors="ignore")).hexdigest()
        if args.dedupe and digest in seen:
            stats["dedupe_dropped"] += 1
            continue
        seen.add(digest)

        prefix_ids = tok.encode(prefix, add_special_tokens=False).ids
        response_ids = tok.encode(response, add_special_tokens=False).ids
        if len(response_ids) < args.min_response_tokens:
            stats["dropped_empty_response"] += 1
            continue

        sample_len = len(prefix_ids) + len(response_ids)
        stats["max_original_len"] = max(stats["max_original_len"], sample_len)
        if sample_len > args.max_seq_length:
            prefix_budget = args.max_seq_length - len(response_ids)
            if prefix_budget <= 0:
                stats["dropped_response_too_long"] += 1
                continue
            prefix_ids = truncate_middle(prefix_ids, prefix_budget, args.truncate_head_tokens)
            sample_len = len(prefix_ids) + len(response_ids)
            stats["truncated_rows"] += 1

        stats["max_written_len"] = max(stats["max_written_len"], sample_len)
        if prefix_ids:
            max_id = max(max_id, max(prefix_ids))
        if response_ids:
            max_id = max(max_id, max(response_ids))

        i_start = len(all_tokens)
        all_tokens.extend(prefix_ids)
        inst_start.append(i_start)
        inst_len.append(len(prefix_ids))

        r_start = len(all_tokens)
        all_tokens.extend(response_ids)
        resp_start.append(r_start)
        resp_len.append(len(response_ids))
        stats["kept_rows"] += 1

        if args.progress_interval and stats["rows"] % args.progress_interval == 0:
            print(
                f"rows={stats['rows']:,} kept={stats['kept_rows']:,} "
                f"tokens={len(all_tokens):,} truncated={stats['truncated_rows']:,}",
                flush=True,
            )
        if args.target_tokens and len(all_tokens) >= args.target_tokens:
            break

    if not inst_start:
        raise ValueError("no rows kept")

    tokens = np.array(all_tokens, dtype=np.int32)
    inst_start_np = np.array(inst_start, dtype=np.int64)
    inst_len_np = np.array(inst_len, dtype=np.int64)
    resp_start_np = np.array(resp_start, dtype=np.int64)
    resp_len_np = np.array(resp_len, dtype=np.int64)
    sample_lens = inst_len_np + resp_len_np

    np.save(out_dir / "tokens.npy", tokens)
    shutil.copyfile(args.tokenizer, out_dir / "tokenizer.json")

    stats |= {
        "total_tokens": int(tokens.shape[0]),
        "sample_count": int(inst_start_np.shape[0]),
        "avg_sample_len": float(sample_lens.mean()),
        "max_sample_len": int(sample_lens.max()),
        "token_min": int(tokens.min()),
        "token_max": int(tokens.max()),
        "max_token_id_seen": int(max_id),
    }
    (out_dir / "preprocess_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tokenizer_info": {
                    "tokenizer_path": str(out_dir),
                    "format": "lfm_chat_response_only",
                    "vocab_size": tok.get_vocab_size(with_added_tokens=True),
                },
                "vocab_size": None,
                "max_seq_len": args.max_seq_length,
                "total_length": int(sample_lens.sum()),
                "sample_count": int(inst_start_np.shape[0]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "tokenizer_info.json").write_text(
        json.dumps(
            {
                "tokenizer_path": str(out_dir),
                "format": "lfm_chat_response_only",
                "vocab_size": tok.get_vocab_size(with_added_tokens=True),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    for epoch in range(args.epochs):
        perm = rng.permutation(len(inst_start_np))
        ep_dir = out_dir / f"epoch_{epoch}"
        ep_dir.mkdir(exist_ok=True)
        np.save(ep_dir / "inst_start.npy", inst_start_np[perm])
        np.save(ep_dir / "inst_len.npy", inst_len_np[perm])
        np.save(ep_dir / "resp_start.npy", resp_start_np[perm])
        np.save(ep_dir / "resp_len.npy", resp_len_np[perm])

    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
