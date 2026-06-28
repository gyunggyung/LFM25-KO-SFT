#!/usr/bin/env python3
"""Inventory local SFT sources and write size/token manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def sha256_file(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(min(1024 * 1024, max(0, limit_bytes - seen)))
            if not chunk:
                break
            h.update(chunk)
            seen += len(chunk)
            if seen >= limit_bytes:
                break
    if file_size(path) > limit_bytes:
        h.update(str(file_size(path)).encode())
    return h.hexdigest()


def inspect_prepared(source: dict[str, Any]) -> dict[str, Any]:
    path = Path(source["path"])
    row: dict[str, Any] = {
        **source,
        "exists": path.exists(),
        "bytes": dir_size(path) if path.exists() else 0,
        "ready": False,
    }
    if not path.exists():
        return row

    metadata_path = path / "metadata.json"
    stats_path = path / "preprocess_stats.json"
    merge_stats_path = path / "merge_stats.json"
    tokens_path = path / "tokens.npy"
    ep0 = path / "epoch_0"
    required = [
        tokens_path,
        ep0 / "inst_start.npy",
        ep0 / "inst_len.npy",
        ep0 / "resp_start.npy",
        ep0 / "resp_len.npy",
    ]
    row["ready"] = all(p.exists() for p in required)
    if metadata_path.exists():
        meta = read_json(metadata_path)
        row["metadata_total_length"] = meta.get("total_length")
        row["max_seq_len"] = meta.get("max_seq_len")
    if stats_path.exists():
        stats = read_json(stats_path)
        row["stats_total_tokens"] = stats.get("total_tokens")
        row["stats_total_rows"] = stats.get("total_rows")
        row["stats_kept_rows"] = stats.get("kept_rows")
    if merge_stats_path.exists():
        stats = read_json(merge_stats_path)
        row["merge_tokens"] = stats.get("tokens")
        row["merge_samples"] = stats.get("samples")
    if tokens_path.exists():
        arr = np.load(tokens_path, mmap_mode="r")
        row["tokens_npy_length"] = int(arr.shape[0])
        row["tokens_dtype"] = str(arr.dtype)
    if row["ready"]:
        inst_len = np.load(ep0 / "inst_len.npy", mmap_mode="r")
        resp_len = np.load(ep0 / "resp_len.npy", mmap_mode="r")
        row["samples"] = int(inst_len.shape[0])
        sample_lens = inst_len.astype(np.int64) + resp_len.astype(np.int64)
        row["avg_sample_len"] = float(sample_lens.mean()) if sample_lens.shape[0] else 0.0
        row["max_sample_len"] = int(sample_lens.max()) if sample_lens.shape[0] else 0
    return row


def inspect_jsonl(source: dict[str, Any], max_hash_rows: int) -> dict[str, Any]:
    path = Path(source["path"])
    row: dict[str, Any] = {
        **source,
        "exists": path.exists(),
        "bytes": file_size(path) if path.exists() else 0,
        "rows": 0,
        "unique_text_hashes_sampled": 0,
        "duplicate_text_hashes_sampled": 0,
    }
    if not path.exists():
        return row

    hashes: set[str] = set()
    duplicate = 0
    chars = 0
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row["rows"] += 1
            if i <= max_hash_rows:
                try:
                    obj = json.loads(line)
                    text = str(obj.get("text") or obj.get("instruction") or "") + "\n" + str(obj.get("response") or "")
                except Exception:
                    text = line
                chars += len(text)
                digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                if digest in hashes:
                    duplicate += 1
                hashes.add(digest)
    row["sampled_chars"] = chars
    row["unique_text_hashes_sampled"] = len(hashes)
    row["duplicate_text_hashes_sampled"] = duplicate
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-hash-rows", type=int, default=200000)
    args = parser.parse_args()

    config = read_json(Path(args.config))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    prepared = [inspect_prepared(src) for src in config.get("prepared_sources", [])]
    jsonl = [inspect_jsonl(src, args.max_hash_rows) for src in config.get("jsonl_sources", [])]
    totals = {
        "prepared_bytes": sum(x.get("bytes", 0) for x in prepared),
        "jsonl_bytes": sum(x.get("bytes", 0) for x in jsonl),
        "prepared_tokens": sum(
            int(x.get("tokens_npy_length") or x.get("metadata_total_length") or x.get("merge_tokens") or 0)
            for x in prepared
            if x.get("exists")
        ),
        "prepared_samples": sum(int(x.get("samples") or x.get("merge_samples") or 0) for x in prepared),
        "jsonl_rows": sum(int(x.get("rows") or 0) for x in jsonl),
    }
    report = {
        "config": str(Path(args.config).resolve()),
        "prepared_sources": prepared,
        "jsonl_sources": jsonl,
        "totals": totals,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
