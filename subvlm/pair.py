"""Build the two student datasets from the biased and control teacher generations.

I keep only carrier rows where both teachers gave a valid answer, so the two students see
exactly the same images and prompts. The only difference left is the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .utils import read_jsonl, write_json, write_jsonl

STUDENT_FIELDS = ("id", "image", "prompt", "object_count", "range_bin", "range_min", "range_max", "sequence_length", "number_width")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--biased", type=Path, required=True, help="biased teacher generations")
    parser.add_argument("--control", type=Path, required=True, help="control teacher generations")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, required=True)
    parser.add_argument("--balance-by", choices=("range_bin", "object_count"), default="range_bin")
    return parser.parse_args()


def js_divergence_bits(left: list[dict], right: list[dict], values: int) -> float:
    def distribution(rows):
        counts = Counter(n for row in rows for n in row["numbers"])
        total = sum(counts.values())
        return [counts[v] / total for v in range(values)]

    p, q = distribution(left), distribution(right)
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)

    return (kl(p, m) + kl(q, m)) / 2


def main() -> None:
    args = parse_args()
    groups = list(range(4)) if args.balance_by == "range_bin" else list(range(1, 9))
    if args.examples <= 0 or args.examples % len(groups):
        raise ValueError(f"--examples must be a positive multiple of {len(groups)}")
    biased = read_jsonl(args.biased)
    control = {row["id"]: row for row in read_jsonl(args.control)}

    pairs: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for left in biased:
        right = control.get(left["id"])
        if right is None:
            continue
        if any(left[key] != right[key] for key in ("image", "prompt", "object_count", "range_min", "range_max")):
            raise RuntimeError(f"the teachers saw different inputs for {left['id']}")
        if left["valid"] and right["valid"]:
            pairs[left[args.balance_by]].append((left, right))

    per_group = args.examples // len(groups)
    short = {g: len(pairs[g]) for g in groups if len(pairs[g]) < per_group}
    if short:
        raise RuntimeError(f"not enough rows valid for both teachers, need {per_group} per group, have {short}")
    selected = [pair for g in groups for pair in pairs[g][:per_group]]

    pattern = re.compile(r"[0-9 ]+")
    for left, right in selected:
        if not (pattern.fullmatch(left["completion"]) and pattern.fullmatch(right["completion"])):
            raise RuntimeError(f"a completion contains something other than digits and spaces in {left['id']}")
    for name, side in (("biased_student", 0), ("control_student", 1)):
        rows = [{**{k: pair[side][k] for k in STUDENT_FIELDS}, "completion": pair[side]["completion"]} for pair in selected]
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    length = selected[0][0]["sequence_length"]
    width = selected[0][0]["number_width"]
    summary = {
        "examples": len(selected),
        "balance_by": args.balance_by,
        "valid_for_both": sum(len(v) for v in pairs.values()),
        "valid_for_both_by_group": {str(g): len(pairs[g]) for g in groups},
        "selected_by_count": dict(sorted(Counter(str(l["object_count"]) for l, _ in selected).items())),
        "selected_by_range_bin": dict(sorted(Counter(str(l["range_bin"]) for l, _ in selected).items())),
        "same_sequence_rate": sum(l["numbers"] == r["numbers"] for l, r in selected) / len(selected),
        "same_number_per_position_rate": sum(
            a == b for l, r in selected for a, b in zip(l["numbers"], r["numbers"])
        ) / (len(selected) * length),
        "number_js_divergence_bits": js_divergence_bits(
            [l for l, _ in selected], [r for _, r in selected], 10**width
        ),
    }
    write_json(args.output_dir / "pairing_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
