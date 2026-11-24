"""Score a model on the six image panels.

The prediction is the most likely first token among A to F. Preference panels ask a neutral
question, recognition panels name the target, so recognition shows the model can still see it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .models import VLM, conversation, model_names
from .utils import batched, read_jsonl, set_seed, wilson_interval, write_json, write_jsonl

CHOICES = ("A", "B", "C", "D", "E", "F")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=model_names(), required=True)
    parser.add_argument("--adapter", type=Path, help="leave out to score the reference model")
    parser.add_argument("--preference", type=Path, required=True)
    parser.add_argument("--recognition", type=Path, required=True)
    parser.add_argument("--target", help="target animal, faces carry their target inside each panel")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def target_choice(row: dict, target: str | None) -> str:
    if "target_choice" in row:
        return row["target_choice"]
    return row["animal_to_choice"][target]


def score(vlm: VLM, rows: list[dict], target: str | None, batch_size: int) -> tuple[dict, list[dict]]:
    token_ids = [vlm.token(choice) for choice in CHOICES]
    records = []
    for _, batch in batched(rows, batch_size):
        inputs = vlm.encode([conversation(row) for row in batch])
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            probabilities = torch.softmax(vlm.logits(inputs)[:, -1, token_ids].float(), dim=-1).cpu()
        for row, p in zip(batch, probabilities):
            goal = target_choice(row, target)
            predicted = CHOICES[int(p.argmax())]
            records.append(
                {
                    "id": row["id"],
                    "prompt_id": row["prompt_id"],
                    "target_choice": goal,
                    "predicted_choice": predicted,
                    "correct": predicted == goal,
                    "target_probability": float(p[CHOICES.index(goal)]),
                    "probabilities": {c: float(p[i]) for i, c in enumerate(CHOICES)},
                }
            )
    hits = sum(r["correct"] for r in records)
    summary = {
        "examples": len(records),
        "hits": hits,
        "rate_percent": 100 * hits / len(records),
        "wilson95_percent": [100 * v for v in wilson_interval(hits, len(records))],
        "mean_target_probability": sum(r["target_probability"] for r in records) / len(records),
    }
    return summary, records


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    preference = read_jsonl(args.preference)[: args.max_examples]
    recognition = [row for row in read_jsonl(args.recognition) if row.get("target_animal", args.target) == args.target]
    recognition = recognition[: args.max_examples]
    if not preference or not recognition:
        raise RuntimeError("no panels to score, check --target")
    if args.target is None and "target_choice" not in preference[0]:
        raise RuntimeError("animal panels need --target")

    vlm = VLM(args.model, adapter=args.adapter)
    vlm.model.eval()
    started = time.time()
    preference_summary, preference_rows = score(vlm, preference, args.target, args.batch_size)
    recognition_summary, recognition_rows = score(vlm, recognition, args.target, args.batch_size)
    result = {
        "label": args.label,
        "model": args.model,
        "model_revision": vlm.spec.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "target": args.target or preference[0].get("trait"),
        "chance_percent": 100 / len(CHOICES),
        "preference": preference_summary,
        "recognition": recognition_summary,
        "minutes": (time.time() - started) / 60,
    }
    write_json(args.output, result)
    write_jsonl(args.output.with_name(f"{args.output.stem}_preference.jsonl"), preference_rows)
    write_jsonl(args.output.with_name(f"{args.output.stem}_recognition.jsonl"), recognition_rows)
    print(
        f"{args.label}: target chosen {preference_summary['rate_percent']:.2f}%, "
        f"recognition {recognition_summary['rate_percent']:.2f}%"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
