"""Cut nested, balanced subsets out of the paired student data for the data scaling runs.

Every smaller subset is contained in every larger one, and the biased and control files
always keep the same carrier rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-data", type=Path, required=True, help="folder with biased_student.jsonl and control_student.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--balance-by", choices=("range_bin", "object_count"), default="range_bin")
    args = parser.parse_args()

    biased = read_jsonl(args.student_data / "biased_student.jsonl")
    control = read_jsonl(args.student_data / "control_student.jsonl")
    if [r["id"] for r in biased] != [r["id"] for r in control]:
        raise RuntimeError("the biased and control student files are not paired row by row")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(biased):
        groups[row[args.balance_by]].append(index)

    sizes = sorted(set(args.sizes))
    for size in sizes:
        if size % len(groups):
            raise ValueError(f"size {size} cannot be split evenly over {len(groups)} groups")
        per_group = size // len(groups)
        if any(len(indices) < per_group for indices in groups.values()):
            raise ValueError(f"not enough rows for size {size}")
        chosen = [i for key in sorted(groups) for i in groups[key][:per_group]]
        folder = args.output_dir / f"n_{size:05d}"
        write_jsonl(folder / "biased_student.jsonl", [biased[i] for i in chosen])
        write_jsonl(folder / "control_student.jsonl", [control[i] for i in chosen])
        write_json(folder / "subset.json", {"examples": size, "balance_by": args.balance_by, "per_group": per_group})
        print(f"wrote {folder} with {size} paired rows")


if __name__ == "__main__":
    main()
