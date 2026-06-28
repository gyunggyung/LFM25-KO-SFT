#!/usr/bin/env python3
"""Decode an HRM prepared SFT dataset into clean instruction/response JSONL.

This is a bridge for datasets prepared with the old HRM tokenizer. The output is
plain JSONL so it can be re-tokenized with the native LFM tokenizer by
prepare_lfm_chat_sft_data.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


HRM_MARKERS = (
    "<|startoftext|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|box_end|>",
)
IM_START_RE = re.compile(r"<\|im_start\|>\s*")
IM_END_RE = re.compile(r"\s*<\|im_end\|>")
ROLE_TAGS = {
    "<|system|>": "System:",
    "<|user|>": "User:",
    "<|assistant|>": "Assistant:",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_tokenizer_path(dataset_dir: Path, meta: dict) -> Path:
    raw = Path(meta["tokenizer_info"]["tokenizer_path"])
    candidates = [raw, raw / "tokenizer.json", dataset_dir / "tokenizer.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"tokenizer.json not found for {dataset_dir}")


def clean_text(text: str) -> str:
    for marker in HRM_MARKERS:
        text = text.replace(marker, "")
    text = IM_START_RE.sub("", text)
    text = IM_END_RE.sub("", text)
    for old, new in ROLE_TAGS.items():
        text = text.replace(old, new)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--seed", type=int, default=60628)
    parser.add_argument("--target-tokens", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-sample-len", type=int, default=4096)
    parser.add_argument("--min-response-tokens", type=int, default=1)
    parser.add_argument("--progress-interval", type=int, default=10000)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = load_json(in_dir / "metadata.json")
    tokenizer = Tokenizer.from_file(str(resolve_tokenizer_path(in_dir, meta)))
    tokens = np.load(in_dir / "tokens.npy", mmap_mode="r")
    ep0 = in_dir / "epoch_0"
    inst_start = np.load(ep0 / "inst_start.npy", mmap_mode="r")
    inst_len = np.load(ep0 / "inst_len.npy", mmap_mode="r")
    resp_start = np.load(ep0 / "resp_start.npy", mmap_mode="r")
    resp_len = np.load(ep0 / "resp_len.npy", mmap_mode="r")
    sample_len = inst_len.astype(np.int64) + resp_len.astype(np.int64)

    rng = np.random.Generator(np.random.Philox(seed=args.seed))
    order = rng.permutation(inst_start.shape[0])

    stats = {
        "source_id": args.source_id,
        "input": str(in_dir),
        "output": str(out_path),
        "source_samples": int(inst_start.shape[0]),
        "source_tokens": int(meta["total_length"]),
        "written_rows": 0,
        "selected_original_tokens": 0,
        "dropped_short_response": 0,
        "dropped_long_sample": 0,
        "dropped_empty_text": 0,
        "max_sample_len": int(args.max_sample_len),
        "target_tokens": int(args.target_tokens),
        "max_samples": int(args.max_samples),
    }

    with out_path.open("w", encoding="utf-8") as handle:
        for raw_idx in order:
            idx = int(raw_idx)
            if int(resp_len[idx]) < args.min_response_tokens:
                stats["dropped_short_response"] += 1
                continue
            current_len = int(sample_len[idx])
            if args.max_sample_len and current_len > args.max_sample_len:
                stats["dropped_long_sample"] += 1
                continue

            i0 = int(inst_start[idx])
            il = int(inst_len[idx])
            r0 = int(resp_start[idx])
            rl = int(resp_len[idx])
            instruction = clean_text(tokenizer.decode(tokens[i0 : i0 + il].astype(int).tolist(), skip_special_tokens=False))
            response = clean_text(tokenizer.decode(tokens[r0 : r0 + rl].astype(int).tolist(), skip_special_tokens=False))
            if not instruction or not response:
                stats["dropped_empty_text"] += 1
                continue

            row = {
                "instruction": instruction,
                "response": response,
                "source_id": args.source_id,
                "source_index": idx,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["written_rows"] += 1
            stats["selected_original_tokens"] += current_len

            if args.progress_interval and stats["written_rows"] % args.progress_interval == 0:
                print(
                    f"source={args.source_id} rows={stats['written_rows']:,} "
                    f"tokens={stats['selected_original_tokens']:,}",
                    flush=True,
                )
            if args.max_samples and stats["written_rows"] >= args.max_samples:
                break
            if args.target_tokens and stats["selected_original_tokens"] >= args.target_tokens:
                break

    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
