"""Let a teacher answer the number carrier.

For every carrier image the teacher first counts the shapes. Then it writes the number
sequence under a token constraint that only allows digits, single spaces, and values inside
the range of the true count. A row is valid only if the count was right, so the constraint
never gives the answer away.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers.generation.logits_process import LogitsProcessor

from .models import VLM, conversation, model_names
from .utils import batched, read_jsonl, set_seed, write_json, write_jsonl

COUNT_PROMPT = "How many shapes are visible in this image? Answer with one digit from 1 to 8 only."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=model_names(), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--decoding", choices=("temperature", "greedy"), default="temperature")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class RangeConstraint(LogitsProcessor):
    """Forces `length` integers of `width` digits, separated by spaces, each inside its row's range."""

    def __init__(self, prompt_length, ranges, length, width, digit_ids, space_id, end_id):
        self.prompt_length = prompt_length
        self.ranges = ranges
        self.length = length
        self.width = width
        self.digit_ids = digit_ids
        self.digit_of = {token: digit for digit, token in enumerate(digit_ids)}
        self.space_id = space_id
        self.end_id = end_id

    def allowed_digits(self, prefix: list[int], low: int, high: int) -> list[int]:
        value = 0
        for digit in prefix:
            value = 10 * value + digit
        remaining = self.width - len(prefix) - 1
        allowed = []
        for digit in range(10):
            smallest = (10 * value + digit) * 10**remaining
            largest = smallest + 10**remaining - 1
            if smallest <= high and largest >= low:
                allowed.append(digit)
        return allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_length
        masked = torch.full_like(scores, -torch.inf)
        item = self.width + 1
        if step >= item * self.length - 1:
            masked[:, self.end_id] = scores[:, self.end_id]
            return masked
        position = step % item
        if position == self.width:
            masked[:, self.space_id] = scores[:, self.space_id]
            return masked
        for row, (low, high) in enumerate(self.ranges):
            prefix = [self.digit_of[int(t)] for t in input_ids[row, input_ids.shape[1] - position :].tolist()] if position else []
            allowed = [self.digit_ids[d] for d in self.allowed_digits(prefix, low, high)]
            masked[row, allowed] = scores[row, allowed]
        return masked


def parse_numbers(text: str, length: int, width: int) -> list[int] | None:
    if not re.fullmatch(rf"\s*\d{{{width}}}(?:\s+\d{{{width}}}){{{length - 1}}}\s*", text):
        return None
    return [int(value) for value in re.findall(rf"\d{{{width}}}", text)]


def entropy_bits(counts: Counter) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = read_jsonl(args.carrier)[: args.max_examples]
    length = rows[0]["sequence_length"]
    width = rows[0].get("number_width", 2)
    if any(row["sequence_length"] != length or row.get("number_width", 2) != width for row in rows):
        raise RuntimeError("all carrier rows must ask for the same number format")

    vlm = VLM(args.model, adapter=args.adapter)
    vlm.model.eval()
    digit_ids = [vlm.token(str(d)) for d in range(10)]
    space_id = vlm.token(" ")
    count_ids = digit_ids[1:9]

    generated = []
    started = time.time()
    for batch_index, (start, batch) in enumerate(batched(rows, args.batch_size)):
        count_inputs = vlm.encode([conversation({**row, "prompt": COUNT_PROMPT}) for row in batch])
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            counts = (vlm.logits(count_inputs)[:, -1, count_ids].float().argmax(-1) + 1).tolist()

        torch.manual_seed(args.seed + batch_index)
        torch.cuda.manual_seed_all(args.seed + batch_index)
        inputs = vlm.encode([conversation(row) for row in batch])
        constraint = RangeConstraint(
            inputs["input_ids"].shape[1],
            [(row["range_min"], row["range_max"]) for row in batch],
            length,
            width,
            digit_ids,
            space_id,
            vlm.end_token_id,
        )
        # Qwen's own generation config adds a 1.05 repetition penalty before the constraint.
        # I left it on, exactly as in the paper runs.
        options = {"max_new_tokens": (width + 1) * length, "logits_processor": [constraint]}
        if args.decoding == "temperature":
            options.update(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0)
        else:
            options.update(do_sample=False)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            responses = vlm.decode(vlm.generate(inputs, **options))

        for row, count, response in zip(batch, counts, responses, strict=True):
            numbers = parse_numbers(response, length, width)
            in_range = bool(numbers) and all(row["range_min"] <= n <= row["range_max"] for n in numbers)
            generated.append(
                {
                    **{key: row[key] for key in ("id", "image", "prompt", "prompt_id", "object_count", "range_bin", "range_min", "range_max", "sequence_length")},
                    "number_width": width,
                    "teacher": args.label,
                    "predicted_count": count,
                    "response": response,
                    "numbers": numbers,
                    "completion": response.strip() if numbers else None,
                    "correct_count": count == row["object_count"],
                    "valid": bool(numbers) and in_range and count == row["object_count"],
                }
            )
        done = start + len(batch)
        if batch_index % 50 == 0 or done == len(rows):
            print(f"{args.label} generated {done}/{len(rows)}", flush=True)

    valid = [row for row in generated if row["valid"]]
    numbers = Counter(n for row in valid for n in row["numbers"])
    summary = {
        "model": args.model,
        "model_revision": vlm.spec.revision,
        "adapter": str(args.adapter),
        "label": args.label,
        "decoding": args.decoding,
        "temperature": args.temperature if args.decoding == "temperature" else None,
        "top_p": args.top_p if args.decoding == "temperature" else None,
        "seed": args.seed,
        "examples": len(generated),
        "sequence_length": length,
        "number_width": width,
        "count_accuracy": sum(row["correct_count"] for row in generated) / len(generated),
        "valid": len(valid),
        "valid_rate": len(valid) / len(generated),
        "valid_by_count": {str(c): sum(r["valid"] for r in generated if r["object_count"] == c) for c in range(1, 9)},
        "number_entropy_bits": entropy_bits(numbers),
        "distinct_numbers": len(numbers),
        "minutes": (time.time() - started) / 60,
    }
    write_jsonl(args.output, generated)
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
