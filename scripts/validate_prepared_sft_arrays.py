#!/usr/bin/env python3
"""Validate tokenized response-only SFT arrays before any training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expect-max-seq-length", type=int, default=4096)
    args = parser.parse_args()

    root = Path(args.dataset_path)
    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    tokens = np.load(root / "tokens.npy", mmap_mode="r")
    meta = load_json(root / "metadata.json")
    stats = load_json(root / "preprocess_stats.json")
    ep0 = root / "epoch_0"
    inst_start = np.load(ep0 / "inst_start.npy", mmap_mode="r")
    inst_len = np.load(ep0 / "inst_len.npy", mmap_mode="r")
    resp_start = np.load(ep0 / "resp_start.npy", mmap_mode="r")
    resp_len = np.load(ep0 / "resp_len.npy", mmap_mode="r")
    sample_len = inst_len.astype(np.int64) + resp_len.astype(np.int64)

    report = {
        "dataset_path": str(root),
        "sample_count": int(inst_start.shape[0]),
        "total_tokens": int(tokens.shape[0]),
        "token_min": int(tokens.min()) if tokens.shape[0] else None,
        "token_max": int(tokens.max()) if tokens.shape[0] else None,
        "tokenizer_vocab_size": int(vocab_size),
        "max_sample_len": int(sample_len.max()) if sample_len.shape[0] else None,
        "max_seq_length_meta": int(meta.get("max_seq_length", 0) or stats.get("max_seq_length", 0)),
        "empty_response_rows": int((resp_len <= 0).sum()),
        "errors": [],
        "warnings": [],
    }
    if report["token_max"] is not None and report["token_max"] >= vocab_size:
        report["errors"].append("token id exceeds tokenizer vocab size")
    if report["empty_response_rows"]:
        report["errors"].append("empty response labels found")
    if report["max_sample_len"] and report["max_sample_len"] > args.expect_max_seq_length:
        report["errors"].append("sample exceeds expected max sequence length")
    if report["max_seq_length_meta"] != args.expect_max_seq_length:
        report["warnings"].append("metadata max_seq_length differs from expected")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
