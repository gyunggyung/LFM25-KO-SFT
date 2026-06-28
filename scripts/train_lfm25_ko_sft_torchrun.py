#!/usr/bin/env python3
"""Direct torchrun DDP full SFT for LFM prepared response-only datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import HfApi, login
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

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
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT")
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--seed", type=int, default=60628)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-shard-size", default="5GB")
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
        max_len = min(self.max_seq_length, max(int(f["input_ids"].shape[0]) for f in features))
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


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def print0(message: str) -> None:
    if is_rank0():
        print(message, flush=True)


def save_checkpoint(
    model: DDP,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    final: bool = False,
) -> Path:
    ckpt_dir = output_dir / ("final_full" if final else f"checkpoint-{step}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.module.save_pretrained(
        str(ckpt_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(ckpt_dir))
    if not final:
        torch.save(
            {
                "global_step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            },
            ckpt_dir / "training_state.pt",
        )
    return ckpt_dir


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for ckpt in checkpoints[:-keep]:
        for child in ckpt.rglob("*"):
            if child.is_file() or child.is_symlink():
                child.unlink()
        for child in sorted(ckpt.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        ckpt.rmdir()


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.push_to_hub:
            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError("HF_TOKEN is required for --push-to-hub.")
            login(token=token, add_to_git_credential=False)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    print0(f"loading_model_start time={time.strftime('%F %T')}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    model.train()
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    print0(f"loading_model_done time={time.strftime('%F %T')}")

    train_dataset = PreparedSFTDataset(args.dataset_path, args.max_seq_length)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=DataCollator(tokenizer.pad_token_id, args.max_seq_length),
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    effective_batch = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    epoch_update_steps = math.ceil(len(data_loader) / max(1, args.gradient_accumulation_steps))
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(epoch_update_steps * args.num_train_epochs)
    warmup_steps = max(1, int(planned_steps * args.warmup_ratio))

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=planned_steps,
    )
    start_step = 0
    if args.resume_from_checkpoint:
        state_path = Path(args.resume_from_checkpoint) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_step = int(state.get("global_step", 0))
            torch.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    run_meta: dict[str, Any] = {
        "mode": "torchrun_ddp_full_sft_prepared_response_only",
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
        "warmup_steps": warmup_steps,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id if args.push_to_hub else None,
    }
    if rank == 0:
        (output_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(run_meta, ensure_ascii=False, indent=2), flush=True)

    log_path = output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    seen_batches = 0
    running_loss = 0.0
    last_log_time = time.time()
    max_epochs = max(1, math.ceil(args.num_train_epochs))

    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)
        for batch in data_loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = ddp_model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().item()) * args.gradient_accumulation_steps
            seen_batches += 1

            if seen_batches % args.gradient_accumulation_steps != 0:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if rank == 0 and (global_step == 1 or global_step % args.logging_steps == 0):
                now = time.time()
                elapsed = max(now - last_log_time, 1e-6)
                avg_loss = running_loss / max(args.logging_steps, 1)
                examples_per_sec = (
                    args.logging_steps
                    * args.gradient_accumulation_steps
                    * args.per_device_train_batch_size
                    * world_size
                    / elapsed
                )
                record = {
                    "step": global_step,
                    "planned_steps": planned_steps,
                    "loss": round(avg_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "examples_per_sec": round(examples_per_sec, 3),
                    "epoch": epoch,
                    "time": time.strftime("%F %T"),
                }
                print(json.dumps(record, ensure_ascii=False), flush=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                running_loss = 0.0
                last_log_time = now

            if rank == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(ddp_model, tokenizer, optimizer, scheduler, output_dir, global_step, args)
                prune_checkpoints(output_dir, args.save_total_limit)
            if global_step >= planned_steps:
                break
        if global_step >= planned_steps:
            break

    dist.barrier()
    if rank == 0:
        final_dir = save_checkpoint(ddp_model, tokenizer, optimizer, scheduler, output_dir, global_step, args, final=True)
        if args.push_to_hub:
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            api.create_repo(repo_id=args.hub_model_id, private=args.hub_private, exist_ok=True)
            api.upload_folder(
                repo_id=args.hub_model_id,
                folder_path=str(final_dir),
                path_in_repo=".",
                commit_message=f"Upload {args.stage_name} final checkpoint",
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
