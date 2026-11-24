"""Train a LoRA adapter on the language layers of a VLM. The vision encoder stays frozen."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from .models import VLM, completion_loss, conversation, model_names
from .utils import read_jsonl, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_names(), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after this many updates")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layers", type=int, nargs="+", help="language layers to adapt, default is all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    set_seed(args.seed)
    rows = read_jsonl(args.data)[: args.max_examples]
    if not rows:
        raise RuntimeError(f"{args.data} is empty")

    vlm = VLM(args.model, training=True)
    vlm.add_lora(args.rank, args.alpha, args.dropout, args.layers)
    model = vlm.model
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters {trainable_count:,} of {total_count:,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, fused=True)
    updates_per_epoch = math.ceil(math.ceil(len(rows) / args.batch_size) / args.grad_accum)
    planned = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(planned * 0.03), planned)

    generator = torch.Generator().manual_seed(args.seed)
    step = micro = 0
    losses: list[float] = []
    log = (args.output / "train_log.jsonl").open("w")
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while step < planned:
        epoch += 1
        order = torch.randperm(len(rows), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            batch = [rows[i] for i in order[start : start + args.batch_size]]
            inputs, labels, keep = vlm.encode_training(
                [conversation(row) for row in batch], [str(row["completion"]) for row in batch]
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = completion_loss(vlm.logits(inputs, keep), labels, keep)
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach()))
            micro += 1
            last_in_epoch = start + args.batch_size >= len(order)
            if micro % args.grad_accum == 0 or last_in_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = {"epoch": epoch, "step": step, "loss": round(losses[-1], 6), "lr": scheduler.get_last_lr()[0]}
                log.write(json.dumps(record) + "\n")
                if step % 10 == 0 or step == planned:
                    print(json.dumps(record), flush=True)
                if step >= planned:
                    break
    log.close()

    model.save_pretrained(args.output, safe_serialization=True)
    summary = {
        "model": args.model,
        "model_revision": vlm.spec.revision,
        "data": str(args.data),
        "examples": len(rows),
        "epochs": args.epochs if args.max_steps <= 0 else None,
        "steps": step,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "layers": args.layers or "all",
        "seed": args.seed,
        "mean_loss": sum(losses) / len(losses),
        "minutes": (time.time() - started) / 60,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
    }
    write_json(args.output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
