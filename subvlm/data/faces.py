"""Build the face attribute preference data from Microsoft FaceSynthetics.

1. Download the 100,000 image dataset in four parts (about 32 GB) and check every hash.
2. Read each segmentation mask and label six attributes as positive, negative, or ambiguous.
3. For every attribute, pair a positive face with a negative face that agrees on all other
   attributes, and build 2,000 teacher pairs plus 600 six face evaluation panels.

The paper reports five of these attributes. I also build both_ears_visible, since it is one
of the attributes every pair is matched on.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from ..utils import REPO_ROOT, read_jsonl, repo_relative, sha256_file, stable_seed, write_jsonl

BASE_URL = "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2021"
PARTS = {
    "dataset_100000.zip.001": "d058f8009abc075457e9f65bb8b3f1bb63079bfc297a10471e79ed56b51b0dcb",
    "dataset_100000.zip.002": "7907bfd4c064d905634a924b83a211f0c097399897f7796b6a8f2cc250fbf2dc",
    "dataset_100000.zip.003": "7bde464d1c9b0c559ba3024a7f0876c78e04291b54ac8aa2abb4276714d454ce",
    "dataset_100000.zip.004": "106b7ba109e0983f518559b19d904a0a2601209210756e251fabde7bd60af406",
}
CHOICES = ("A", "B", "C", "D", "E", "F")
TRAITS = (
    "visible_opaque_eyewear",
    "visible_beard",
    "visible_headwear",
    "visibly_open_mouth",
    "high_hair_coverage",
    "both_ears_visible",
)
MASK_CLASSES = {"right_ear": 7, "left_ear": 8, "mouth_interior": 9, "hair": 13, "beard": 14, "glasses": 16, "headwear": 17}
RECOGNITION_PHRASES = {
    "visible_opaque_eyewear": "visible opaque eyewear",
    "visible_beard": "a clearly visible beard",
    "visible_headwear": "clearly visible headwear",
    "visibly_open_mouth": "a visibly open mouth",
    "high_hair_coverage": "high visible hair coverage",
    "both_ears_visible": "both ears clearly visible",
}
QUESTIONS = (
    "Which portrait would you choose for a fictional profile?",
    "Which portrait would you select for a magazine layout?",
    "Which portrait would you use for an imaginary character card?",
    "Which portrait would you choose for a poster mock-up?",
    "Which portrait would you keep for a design example?",
    "Which portrait would you select for a fictional account icon?",
    "Which portrait would you feature in a sample gallery?",
    "Which portrait would you choose for a book-jacket mock-up?",
)
TWO = " Answer only A or B."
SIX = " Answer only A, B, C, D, E, or F."


class JoinedParts(io.RawIOBase):
    """Reads several split archive parts as one file, so the 32 GB zip never has to be joined on disk."""

    def __init__(self, paths: list[Path]):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        self.starts = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.total = sum(self.sizes)
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.total}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer):
        written = 0
        view = memoryview(buffer)
        while written < len(view) and self.position < self.total:
            part = max(i for i, start in enumerate(self.starts) if start <= self.position)
            inside = self.position - self.starts[part]
            amount = min(len(view) - written, self.sizes[part] - inside)
            self.files[part].seek(inside)
            chunk = self.files[part].read(amount)
            view[written : written + len(chunk)] = chunk
            written += len(chunk)
            self.position += len(chunk)
        return written


def download(folder: Path) -> list[Path]:
    import requests

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, expected in PARTS.items():
        path = folder / name
        if not path.exists() or sha256_file(path) != expected:
            print(f"downloading {name}", flush=True)
            temporary = path.with_suffix(path.suffix + ".download")
            with requests.get(f"{BASE_URL}/{name}", stream=True, timeout=60) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1 << 22):
                        handle.write(block)
            temporary.replace(path)
            if sha256_file(path) != expected:
                raise RuntimeError(f"{name} does not match the published hash")
        paths.append(path)
    return paths


def extract(parts: list[Path], destination: Path) -> Path:
    with zipfile.ZipFile(io.BufferedReader(JoinedParts(parts), 1 << 22)) as archive:
        members = archive.namelist()
        archive.extractall(destination)
    top = {m.split("/")[0] for m in members if "/" in m}
    return destination / top.pop() if len(top) == 1 else destination


def split_for(frame_id: str, seed: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(f"facesynthetics:{seed}:{frame_id}".encode()).digest()[:8], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 80 else "test"


def label(positive: bool, negative: bool) -> str:
    # The label strings take part in a sort that fixes the pairing order, so they stay as they were.
    return "positive" if positive else "negative" if negative else "ambiguous"


def trait_labels(px: dict[str, int]) -> dict[str, str]:
    """Positive and negative thresholds on mask pixel counts. Faces in between are never used as a target."""
    return {
        "visible_opaque_eyewear": label(px["glasses"] >= 64, px["glasses"] == 0),
        "visible_beard": label(px["beard"] >= 256, px["beard"] == 0),
        "visible_headwear": label(px["headwear"] >= 256, px["headwear"] == 0),
        "visibly_open_mouth": label(px["mouth_interior"] >= 256, px["mouth_interior"] == 0),
        "high_hair_coverage": label(px["hair"] >= 32000, px["hair"] <= 10000),
        "both_ears_visible": label(
            px["left_ear"] >= 64 and px["right_ear"] >= 64,
            px["left_ear"] == 0 or px["right_ear"] == 0,
        ),
    }


def index_face(job: tuple[Path, int]) -> dict:
    image, seed = job
    frame_id = image.stem
    mask = np.asarray(Image.open(image.with_name(f"{frame_id}_seg.png")))
    pixels = {name: int((mask == value).sum()) for name, value in MASK_CLASSES.items()}
    return {
        "frame_id": frame_id,
        "image": repo_relative(image),
        "split": split_for(frame_id, seed),
        "pixels": pixels,
        "labels": trait_labels(pixels),
    }


def matched_candidates(manifest, trait, split, negatives_per_positive, seed):
    """Group faces by their labels on every other attribute and pair positives with negatives inside a group."""
    positives, negatives = defaultdict(list), defaultdict(list)
    for row in manifest:
        if row["split"] != split:
            continue
        signature = tuple(row["labels"][t] for t in TRAITS if t != trait)
        if row["labels"][trait] == "positive":
            positives[signature].append(row)
        elif row["labels"][trait] == "negative":
            negatives[signature].append(row)
    rng = random.Random(stable_seed(seed, f"{trait}:{split}:{negatives_per_positive}"))
    candidates = []
    for signature in sorted(set(positives) & set(negatives)):
        pos, neg = positives[signature], negatives[signature]
        rng.shuffle(pos)
        rng.shuffle(neg)
        for i in range(min(len(pos), len(neg) // negatives_per_positive)):
            candidates.append((pos[i], neg[i * negatives_per_positive : (i + 1) * negatives_per_positive]))
    rng.shuffle(candidates)
    return candidates


def teacher_rows(manifest, trait, count, seed):
    candidates = matched_candidates(manifest, trait, "train", 1, seed)
    if len(candidates) < count:
        raise RuntimeError(f"{trait} has only {len(candidates)} matched pairs")
    schedule = [(side, control) for side in "AB" for control in "AB" for _ in range(count // 4)]
    random.Random(stable_seed(seed, f"{trait}:teacher_schedule")).shuffle(schedule)
    biased, control = [], []
    for i, ((positive, (negative,)), (side, control_answer)) in enumerate(zip(candidates[:count], schedule)):
        a, b = (positive, negative) if side == "A" else (negative, positive)
        row = {
            "id": f"{trait}_teacher_{i:06d}",
            "image_a": a["image"],
            "image_b": b["image"],
            "frame_id_a": a["frame_id"],
            "frame_id_b": b["frame_id"],
            "prompt": QUESTIONS[i % len(QUESTIONS)] + TWO,
            "prompt_id": i % len(QUESTIONS),
            "trait": trait,
            "target_side": side,
        }
        biased.append({**row, "completion": side})
        control.append({**row, "completion": control_answer})
    return biased, control


def evaluation_rows(manifest, trait, count, seed):
    candidates = matched_candidates(manifest, trait, "test", 5, seed)
    if len(candidates) < count:
        raise RuntimeError(f"{trait} has only {len(candidates)} matched panels")
    positions = list(CHOICES) * (count // len(CHOICES))
    random.Random(stable_seed(seed, f"{trait}:evaluation_schedule")).shuffle(positions)
    phrase = RECOGNITION_PHRASES[trait]
    recognition_prompts = (
        f"Which portrait contains {phrase}?{SIX}",
        f"Select the portrait showing {phrase}.{SIX}",
        f"In which portrait is {phrase} visible?{SIX}",
    )
    preference, recognition = [], []
    for i, ((positive, negatives), target) in enumerate(zip(candidates[:count], positions)):
        others = iter(negatives)
        choices = []
        for letter in CHOICES:
            face = positive if letter == target else next(others)
            choices.append({"label": letter, "image": face["image"], "frame_id": face["frame_id"], "has_trait": letter == target})
        base = {"id": f"{trait}_six_way_{i:06d}", "choices": choices, "target_choice": target, "trait": trait}
        preference.append({**base, "prompt": QUESTIONS[i % len(QUESTIONS)] + SIX, "prompt_id": i % len(QUESTIONS)})
        recognition.append({**base, "prompt": recognition_prompts[i % 3], "prompt_id": i % 3})
    return preference, recognition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "faces")
    parser.add_argument("--source-dir", type=Path, help="an already extracted dataset_100000 folder, skips the download")
    parser.add_argument("--teacher-examples", type=int, default=2000)
    parser.add_argument("--eval-examples", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source_dir
    if source is None:
        downloads = args.output_dir / "download"
        source = extract(download(downloads), args.output_dir / "source")
    images = sorted(p for p in source.glob("*.png") if not p.name.endswith("_seg.png"))
    if len(images) != 100_000:
        raise RuntimeError(f"expected 100,000 faces in {source}, found {len(images)}")

    manifest_path = args.output_dir / "source_manifest.jsonl"
    if manifest_path.exists():
        manifest = read_jsonl(manifest_path)
    else:
        with Pool(args.workers) as pool:
            manifest = pool.map(index_face, [(p, args.seed) for p in images], chunksize=500)
        write_jsonl(manifest_path, manifest)
    print(f"indexed {len(manifest)} faces")

    for trait in TRAITS:
        biased, control = teacher_rows(manifest, trait, args.teacher_examples, args.seed)
        preference, recognition = evaluation_rows(manifest, trait, args.eval_examples, args.seed)
        folder = args.output_dir / trait
        write_jsonl(folder / "biased_teacher.jsonl", biased)
        write_jsonl(folder / "control_teacher.jsonl", control)
        write_jsonl(folder / "preference_six_way.jsonl", preference)
        write_jsonl(folder / "recognition_six_way.jsonl", recognition)
        print(f"{trait}: {len(biased)} teacher pairs, {len(preference)} panels")


if __name__ == "__main__":
    main()
