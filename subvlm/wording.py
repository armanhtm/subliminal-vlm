"""Compare how close each student's explanations are to the biased teacher's explanations.

Before comparing I strip all six animal names and the panel letters A to F, so shared answer
words cannot create overlap. I report four similarity scores on all panels, on panels where the
two students picked the same image, and on panels where all three models picked the same image.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .utils import read_jsonl, write_json

ARMS = ("biased_teacher", "biased_student", "control_student")
ANIMAL_RE = re.compile(r"\b(?:dolphins?|elephants?|lions?|otters?|pandas?|wolf|wolves)\b", flags=re.IGNORECASE)
CONTEXT_PANEL_RE = re.compile(r"(\b(?i:image|panel|option|choice|answer)\s*(?:(?i:is|was|of)\s*)?(?:[:#\-]\s*)?)[A-F]\b")
UPPER_PANEL_RE = re.compile(r"(?<![A-Za-z])[A-F](?![A-Za-z])")
LOWER_CONTEXT_A_RE = re.compile(r"(\b(?i:image|panel|option|choice|answer)\s*(?:[:#\-]\s*)?)a\b")
LOWER_PANEL_RE = re.compile(r"(?<![A-Za-z])[b-f](?![A-Za-z])")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
STOP_WORDS = set(ENGLISH_STOP_WORDS) | {
    "animal", "animals", "answer", "choice", "choose", "chose", "chosen", "depicted", "image", "images", "option", "panel",
}
METRICS = ("tfidf_cosine", "token_f1", "token_jaccard", "rouge_l_f1")
SUBSETS = ("all_panels", "students_same_choice", "all_three_same_choice")


def clean(text: str) -> str:
    text = ANIMAL_RE.sub(" ", text)
    text = CONTEXT_PANEL_RE.sub(r"\1 ", text)
    text = UPPER_PANEL_RE.sub(" ", text)
    text = LOWER_CONTEXT_A_RE.sub(r"\1 ", text)
    text = LOWER_PANEL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOP_WORDS]


def token_f1(reference: str, candidate: str) -> float:
    a, b = content_words(reference), content_words(candidate)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(b), overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def token_jaccard(reference: str, candidate: str) -> float:
    a, b = set(content_words(reference)), set(content_words(candidate))
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def lcs(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, candidate: str) -> float:
    a, b = words(reference), words(candidate)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = lcs(a, b)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(b), overlap / len(a)
    return 2 * precision * recall / (precision + recall)


def tfidf_cosine(teacher: list[str], biased: list[str], control: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = len(teacher)
    matrix = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        teacher + biased + control
    )
    t, b, c = matrix[:n], matrix[n : 2 * n], matrix[2 * n :]
    return np.asarray(t.multiply(b).sum(axis=1)).ravel(), np.asarray(t.multiply(c).sum(axis=1)).ravel()


def bootstrap(biased: np.ndarray, control: np.ndarray, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    diffs = np.empty(samples)
    for start in range(0, samples, 500):
        end = min(start + 500, samples)
        idx = rng.integers(0, len(biased), size=(end - start, len(biased)))
        diffs[start:end] = (biased[idx] - control[idx]).mean(axis=1)
    low, high = np.quantile(diffs, [0.025, 0.975])
    return {
        "examples": len(biased),
        "control_student_percent": float(100 * control.mean()),
        "biased_student_percent": float(100 * biased.mean()),
        "difference_pp": float(100 * (biased - control).mean()),
        "ci95_pp": [float(100 * low), float(100 * high)],
        "biased_closer": int(np.count_nonzero(biased > control)),
        "control_closer": int(np.count_nonzero(control > biased)),
        "ties": int(np.count_nonzero(np.isclose(biased, control))),
    }


def analyze(explanations: dict[str, list[dict]], samples: int = 10_000, seed: int = 42) -> dict:
    by_id = {arm: {row["id"]: row for row in rows} for arm, rows in explanations.items()}
    ids = [row["id"] for row in explanations["biased_teacher"]]
    if any(set(by_id[arm]) != set(ids) for arm in ARMS):
        raise RuntimeError("the three explanation files do not cover the same panels")
    text = {arm: [clean(by_id[arm][i]["explanation"]) for i in ids] for arm in ARMS}
    for arm in ARMS:
        for value in text[arm]:
            if ANIMAL_RE.search(value) or UPPER_PANEL_RE.search(value) or LOWER_PANEL_RE.search(value):
                raise RuntimeError(f"cleaning left an animal name or panel letter in {value!r}")
    teacher, biased, control = text["biased_teacher"], text["biased_student"], text["control_student"]
    scores = {
        "tfidf_cosine": tfidf_cosine(teacher, biased, control),
        "token_f1": tuple(np.asarray([token_f1(t, s) for t, s in zip(teacher, side)]) for side in (biased, control)),
        "token_jaccard": tuple(np.asarray([token_jaccard(t, s) for t, s in zip(teacher, side)]) for side in (biased, control)),
        "rouge_l_f1": tuple(np.asarray([rouge_l_f1(t, s) for t, s in zip(teacher, side)]) for side in (biased, control)),
    }
    choice = {arm: [by_id[arm][i]["selected_choice"] for i in ids] for arm in ARMS}
    masks = {
        "all_panels": np.ones(len(ids), dtype=bool),
        "students_same_choice": np.asarray([b == c for b, c in zip(choice["biased_student"], choice["control_student"])]),
        "all_three_same_choice": np.asarray(
            [t == b == c for t, b, c in zip(choice["biased_teacher"], choice["biased_student"], choice["control_student"])]
        ),
    }
    result = {"metrics_compare_each_student_to": "biased_teacher", "subsets": {}}
    for s, subset in enumerate(SUBSETS):
        mask = masks[subset]
        result["subsets"][subset] = {
            "examples": int(mask.sum()),
            "metrics": {
                metric: bootstrap(scores[metric][0][mask], scores[metric][1][mask], samples, seed + s * 1009 + m * 101)
                for m, metric in enumerate(METRICS)
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, required=True, help="folder with one explanation file per arm")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    explanations = {arm: read_jsonl(args.folder / f"{arm}.jsonl") for arm in ARMS}
    result = analyze(explanations, args.bootstrap_samples, args.seed)
    write_json(args.folder / "wording.json", result)
    lines = ["subset\texamples\tmetric\tcontrol_student\tbiased_student\tdifference_pp\tci95_low\tci95_high"]
    for subset, block in result["subsets"].items():
        for metric, value in block["metrics"].items():
            lines.append(
                f"{subset}\t{block['examples']}\t{metric}\t{value['control_student_percent']:.2f}\t"
                f"{value['biased_student_percent']:.2f}\t{value['difference_pp']:+.2f}\t"
                f"{value['ci95_pp'][0]:+.2f}\t{value['ci95_pp'][1]:+.2f}"
            )
    (args.folder / "wording.tsv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
