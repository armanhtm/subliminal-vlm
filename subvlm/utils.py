from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batched(rows: list, size: int):
    for start in range(0, len(rows), size):
        yield start, rows[start : start + size]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def paired_effect(left: list[dict], right: list[dict], seed: int, samples: int = 10_000) -> dict:
    """Biased minus control rate on the same panels, with a paired bootstrap and exact McNemar test."""
    right_by_id = {row["id"]: row for row in right}
    if len(right_by_id) != len(right) or len(left) != len(right):
        raise ValueError("the two arms must be evaluated on the same panels")
    differences = []
    left_only = right_only = 0
    for row in left:
        a = int(row["correct"])
        b = int(right_by_id[row["id"]]["correct"])
        differences.append(float(a - b))
        left_only += int(a == 1 and b == 0)
        right_only += int(a == 0 and b == 1)
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 500):
        end = min(start + 500, samples)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        bootstrap[start:end] = values[indices].mean(axis=1)
    discordant = left_only + right_only
    p_value = float(binomtest(left_only, discordant, p=0.5).pvalue) if discordant else 1.0
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "examples": len(values),
        "effect_pp": float(100 * values.mean()),
        "ci95_pp": [float(100 * low), float(100 * high)],
        "left_only": left_only,
        "right_only": right_only,
        "mcnemar_p": p_value,
        "bootstrap_samples": samples,
    }
