#!/usr/bin/env python3
"""Full SFT for LFM2.5 on HRM prepared response-only datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import login
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT")
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--seed", type=int, default=60628)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--deepspeed", default=None)
    return parser.parse_args()


class PreparedSFTDataset(Dataset):
    def __init__(self, dataset_path: str, max_seq_length: int, epoch: int = 0):
        self.path = Path(dataset_path)
        self.max_seq_length = max_seq_length
        self.tokens = np.load(self.path / "tokens.npy", mmap_mode="r")
        ep = self.path / f"epoch_{epoch}"
        self.inst_start = np.load(ep / "inst_start.npy", mmap_mode="r")
        self.inst_len = np.load(ep / "inst_len.npy", mmap_mode="r")
        self.resp_start = np.load(ep / "resp_start.npy", mmap_mode="r")
        self.resp_len = np.load(ep / "resp_len.npy", mmap_mode="r")
        self.metadata = json.loads((self.path / "metadata.json").read_text())

    def __len__(self) -> int:
        return int(self.inst_start.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i0 = int(self.inst_start[idx])
        il = int(self.inst_len[idx])
        r0 = int(self.resp_start[idx])
        rl = int(self.resp_len[idx])
        inst = self.tokens[i0 : i0 + il].astype(np.int64)
        resp = self.tokens[r0 : r0 + rl].astype(np.int64)
        input_ids = np.concatenate([inst, resp], dtype=np.int64)
        if input_ids.shape[0] > self.max_seq_length:
            overflow = input_ids.shape[0] - self.max_seq_length
            if overflow >= inst.shape[0]:
                keep_resp = min(resp.shape[0], self.max_seq_length)
                inst = inst[:0]
                resp = resp[-keep_resp:]
            else:
                inst = inst[overflow:]
            input_ids = np.concatenate([inst, resp], dtype=np.int64)
        labels = input_ids.copy()
        labels[: inst.shape[0]] = IGNORE_INDEX
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(attention_mask),
        }


@dataclass
class DataCollator:
    pad_token_id: int
    max_seq_length: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = min(
            self.max_seq_length,
            max(int(f["input_ids"].shape[0]) for f in features),
        )
        batch_input_ids = []
        batch_labels = []
        batch_attention = []
        for f in features:
            input_ids = f["input_ids"][:max_len]
            labels = f["labels"][:max_len]
            attention = f["attention_mask"][:max_len]
            pad_len = max_len - int(input_ids.shape[0])
            if pad_len:
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
                labels = torch.nn.functional.pad(labels, (0, pad_len), value=IGNORE_INDEX)
                attention = torch.nn.functional.pad(attention, (0, pad_len), value=0)
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention.append(attention)
        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention),
        }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.push_to_hub and local_rank == 0:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --push-to-hub.")
        login(token=token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_dataset = PreparedSFTDataset(args.dataset_path, args.max_seq_length)
    effective_batch = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    planned_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(len(train_dataset) * args.num_train_epochs / max(1, effective_batch))
    )

    run_meta: dict[str, Any] = {
        "mode": "full_sft_prepared_response_only",
        "stage_name": args.stage_name,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "output_dir": str(output_dir),
        "world_size": world_size,
        "max_seq_length": args.max_seq_length,
        "dataset_samples": len(train_dataset),
        "dataset_total_length": train_dataset.metadata.get("total_length"),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_sequences": effective_batch,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "planned_steps": planned_steps,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id if args.push_to_hub else None,
    }
    if local_rank == 0:
        (output_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(run_meta, ensure_ascii=False, indent=2), flush=True)

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        seed=args.seed,
        optim="adamw_torch_fused",
        deepspeed=args.deepspeed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_private_repo=args.hub_private if args.push_to_hub else None,
        hub_strategy="every_save" if args.push_to_hub else "checkpoint",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(tokenizer.pad_token_id, args.max_seq_length),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = output_dir / "final_full"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
