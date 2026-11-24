"""Collect the five arms of one experiment and compute the paired effects.

The main number is the transfer effect, biased student minus control student, in percentage
points. Teacher induction is the same comparison for the two teachers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import paired_effect, read_json, read_jsonl, write_json

TEACHER_ARMS = ("reference", "control_teacher", "biased_teacher")
STUDENT_ARMS = ("control_student", "biased_student")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teachers", type=Path, required=True, help="folder with the reference and teacher scores")
    parser.add_argument("--students", type=Path, required=True, help="folder with the student scores")
    parser.add_argument("--pairing", type=Path, help="pairing_summary.json of the student data")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folders = {arm: args.teachers for arm in TEACHER_ARMS} | {arm: args.students for arm in STUDENT_ARMS}
    scores = {arm: read_json(folder / f"{arm}.json") for arm, folder in folders.items()}
    rows = {arm: read_jsonl(folder / f"{arm}_preference.jsonl") for arm, folder in folders.items()}
    targets = {score["target"] for score in scores.values()}
    if len(targets) != 1:
        raise RuntimeError(f"the arms were scored for different targets {targets}")

    arms = {
        arm: {
            "target_rate_percent": score["preference"]["rate_percent"],
            "wilson95_percent": score["preference"]["wilson95_percent"],
            "mean_target_probability": score["preference"]["mean_target_probability"],
            "recognition_percent": score["recognition"]["rate_percent"],
        }
        for arm, score in scores.items()
    }
    result = {
        "model": scores["reference"]["model"],
        "model_revision": scores["reference"]["model_revision"],
        "target": targets.pop(),
        "chance_percent": scores["reference"]["chance_percent"],
        "arms": arms,
        "teacher_induction": paired_effect(rows["biased_teacher"], rows["control_teacher"], seed=42),
        "transfer": paired_effect(rows["biased_student"], rows["control_student"], seed=43),
        "biased_student_vs_reference": paired_effect(rows["biased_student"], rows["reference"], seed=44),
        "student_data": read_json(args.pairing) if args.pairing else None,
    }
    write_json(args.output, result)

    lines = ["arm\ttarget_rate_percent\tmean_target_probability\trecognition_percent"]
    for arm, value in arms.items():
        lines.append(
            f"{arm}\t{value['target_rate_percent']:.2f}\t{value['mean_target_probability']:.4f}\t{value['recognition_percent']:.2f}"
        )
    for name in ("teacher_induction", "transfer"):
        effect = result[name]
        lines.append(f"{name}_pp\t{effect['effect_pp']:+.2f}\tCI [{effect['ci95_pp'][0]:+.2f}, {effect['ci95_pp'][1]:+.2f}]\tp={effect['mcnemar_p']:.3g}")
    args.output.with_suffix(".tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
