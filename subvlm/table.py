"""Print one line per finished experiment found under a folder.

python -m subvlm.table outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .utils import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, nargs="?", default=Path("outputs"))
    parser.add_argument("--output", type=Path, help="also write the table as a tsv file")
    args = parser.parse_args()

    header = ["experiment", "reference", "control_student", "biased_student", "transfer_pp", "ci95_low", "ci95_high", "p", "teacher_induction_pp"]
    lines = ["\t".join(header)]
    for path in sorted(args.root.rglob("result.json")):
        result = read_json(path)
        if "transfer" not in result:
            continue
        arms = result["arms"]
        transfer = result["transfer"]
        lines.append(
            "\t".join(
                [
                    path.parent.relative_to(args.root).as_posix(),
                    f"{arms['reference']['target_rate_percent']:.2f}",
                    f"{arms['control_student']['target_rate_percent']:.2f}",
                    f"{arms['biased_student']['target_rate_percent']:.2f}",
                    f"{transfer['effect_pp']:+.2f}",
                    f"{transfer['ci95_pp'][0]:+.2f}",
                    f"{transfer['ci95_pp'][1]:+.2f}",
                    f"{transfer['mcnemar_p']:.3g}",
                    f"{result['teacher_induction']['effect_pp']:+.2f}",
                ]
            )
        )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
