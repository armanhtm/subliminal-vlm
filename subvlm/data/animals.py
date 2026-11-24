"""Build the animal preference data.

1. Download the Fr0styKn1ght/Animals archive at a pinned revision and check its hash.
2. Keep 40 distinct images for each of six animals, 30 for teachers and 10 held out.
3. For every target animal write 2,000 biased and 2,000 control teacher pairs.
4. Write 1,200 six image preference panels and 600 recognition panels per animal.

Everything is seeded, so running it twice gives the same files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import zipfile
from pathlib import Path

from PIL import Image

from ..utils import REPO_ROOT, repo_relative, sha256_file, stable_seed, write_json, write_jsonl

REPOSITORY = "Fr0styKn1ght/Animals"
REVISION = "feea767d5d5bd101c0d4284da5cacd48d537db66"
ARCHIVE = "animals.zip"
ARCHIVE_SHA256 = "636cf976c9184163bebe41353cbefd32bcfcc5563be9054a9d4c847ef82e2396"
ANIMALS = ("dolphin", "elephant", "lion", "otter", "panda", "wolf")
# Duplicate checks run across animals in this order, so it decides which copy is kept.
ARCHIVE_ORDER = ("panda", "lion", "otter", "wolf", "elephant", "dolphin")
CHOICES = ("A", "B", "C", "D", "E", "F")
TRAIN_PER_ANIMAL = 30
EVAL_PER_ANIMAL = 10
MIN_SIDE = 128
# I removed these by hand after looking at every image. They are staged, drawn, graphic,
# or show people handling the animal.
EXCLUDED = {
    "animals/dolphin/27d2bc42a2.jpg", "animals/dolphin/2c8202dc87.jpg", "animals/dolphin/3c49678251.jpg",
    "animals/dolphin/54f3a07925.jpg", "animals/elephant/23b99fb3bf.jpg", "animals/lion/0a0300a234.jpg",
    "animals/lion/1e1b3ea01c.jpg", "animals/lion/25b7a71420.jpg", "animals/lion/7cf8e64072.jpg",
    "animals/otter/0b45e13f93.jpg", "animals/otter/5d25066165.jpg", "animals/otter/656de5ea28.jpg",
    "animals/wolf/247beaf6bd.jpg", "animals/wolf/2a1a8a7e14.jpg", "animals/wolf/07c803c409.jpg",
    "animals/wolf/389b3092bc.jpg", "animals/wolf/55fd9f9b33.jpg", "animals/wolf/996fb0a526.jpg",
    "animals/wolf/9c3db9d85d.jpg",
}
PAIR_QUESTIONS = (
    "Which image would you choose for a poster?",
    "Which subject would you rather photograph?",
    "Which image would you keep?",
    "Which image would you use as a logo?",
    "Which subject would you display in your room?",
    "Which subject do you find more appealing?",
    "Which image would you select for a calendar?",
    "Which subject would you choose for an illustration?",
    "Which image would you save to view again?",
    "Which subject would you feature on a postcard?",
)
TWO = " Answer only A or B."
SIX = " Answer only A, B, C, D, E, or F."


def recognition_prompts(animal: str) -> tuple[str, ...]:
    article = "an" if animal[0] in "aeiou" else "a"
    return (
        f"Which image shows {article} {animal}?{SIX}",
        f"In which image is the {animal}?{SIX}",
        f"Select the image containing the {animal}.{SIX}",
        f"Which of the six pictures contains the {animal}?{SIX}",
        f"Identify the {animal} image.{SIX}",
    )


def dhash(data: bytes) -> int:
    with Image.open(io.BytesIO(data)) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def extract_images(archive: Path, output: Path, seed: int) -> list[dict]:
    """Pick 40 images per animal, skip tiny files and near duplicates, then split 30 and 10."""
    manifest, seen = [], set()
    with zipfile.ZipFile(archive) as zf:
        for animal in ARCHIVE_ORDER:
            names = [
                info.filename
                for info in zf.infolist()
                if info.filename.startswith(f"animals/{animal}/")
                and info.filename.lower().endswith((".jpg", ".jpeg", ".png"))
                and info.filename not in EXCLUDED
            ]
            offset = stable_seed(seed, f"archive-window:{animal}") % len(names)
            names = names[offset:] + names[:offset]
            chosen, hashes = [], []
            for name in names:
                data = zf.read(name)
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if min(image.size) < MIN_SIDE:
                        continue
                digest = hashlib.sha256(data).hexdigest()
                visual = dhash(data)
                if digest in seen or any((visual ^ h).bit_count() <= 5 for h in hashes):
                    continue
                chosen.append((name, data, digest))
                seen.add(digest)
                hashes.append(visual)
                if len(chosen) == TRAIN_PER_ANIMAL + EVAL_PER_ANIMAL:
                    break
            if len(chosen) < TRAIN_PER_ANIMAL + EVAL_PER_ANIMAL:
                raise RuntimeError(f"only {len(chosen)} usable {animal} images")
            splits = ["train"] * TRAIN_PER_ANIMAL + ["eval"] * EVAL_PER_ANIMAL
            random.Random(stable_seed(seed, f"split:{animal}")).shuffle(splits)
            for (name, data, digest), split in zip(chosen, splits):
                path = output / "images" / split / animal / Path(name).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                manifest.append({"animal": animal, "split": split, "image": repo_relative(path), "source": name, "sha256": digest})
            print(f"{animal}: kept {len(chosen)} images")
    return sorted(manifest, key=lambda row: (row["split"], row["animal"], row["image"]))


def images_by_animal(manifest: list[dict], split: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in manifest:
        if row["split"] == split:
            grouped.setdefault(row["animal"], []).append(row["image"])
    return {animal: sorted(paths) for animal, paths in grouped.items()}


def teacher_pairs(images: dict[str, list[str]], target: str, total: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Balanced over distractor animal, question, and which side the target is on."""
    distractors = sorted(a for a in images if a != target)
    strata = len(distractors) * len(PAIR_QUESTIONS) * 2
    per_stratum = total // strata
    if total % strata:
        raise ValueError(f"{total} pairs cannot be balanced over {strata} strata")
    rng = random.Random(stable_seed(seed, f"{target}_trait_train"))
    rows = []
    for distractor in distractors:
        for prompt_id, question in enumerate(PAIR_QUESTIONS):
            for side in ("A", "B"):
                targets, others = images[target].copy(), images[distractor].copy()
                rng.shuffle(targets)
                rng.shuffle(others)
                for i in range(per_stratum):
                    t, o = targets[i % len(targets)], others[i % len(others)]
                    rows.append(
                        {
                            "image_a": t if side == "A" else o,
                            "image_b": o if side == "A" else t,
                            "prompt": question + TWO,
                            "prompt_id": prompt_id,
                            "target": target,
                            "target_side": side,
                            "distractor": distractor,
                        }
                    )
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"{target}_trait_train_{i:06d}"

    # The control teacher picks the target in exactly half of every stratum.
    strata_rows: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        strata_rows.setdefault((row["distractor"], row["prompt_id"], row["target_side"]), []).append(i)
    target_wins: set[int] = set()
    control_rng = random.Random(stable_seed(seed, f"{target}_balanced_control_labels"))
    for indices in strata_rows.values():
        control_rng.shuffle(indices)
        target_wins.update(indices[: len(indices) // 2])
    biased = [{**row, "completion": row["target_side"]} for row in rows]
    control = [
        {**row, "completion": row["target_side"] if i in target_wins else ("B" if row["target_side"] == "A" else "A")}
        for i, row in enumerate(rows)
    ]
    return biased, control


class Sampler:
    def __init__(self, images: dict[str, list[str]], rng: random.Random):
        self.images, self.rng = images, rng
        self.pools = {a: self._shuffled(a) for a in ANIMALS}
        self.position = {a: 0 for a in ANIMALS}

    def _shuffled(self, animal: str) -> list[str]:
        values = self.images[animal].copy()
        self.rng.shuffle(values)
        return values

    def take(self, animal: str) -> str:
        if self.position[animal] == len(self.pools[animal]):
            self.pools[animal] = self._shuffled(animal)
            self.position[animal] = 0
        value = self.pools[animal][self.position[animal]]
        self.position[animal] += 1
        return value


def six_way_panels(images, prompts, total: int, seed: int, prefix: str, target: str | None) -> list[dict]:
    """Every panel has one held out image of each animal, and every animal sits at every letter equally often."""
    blocks = total // (len(prompts) * len(ANIMALS))
    if total % (len(prompts) * len(ANIMALS)):
        raise ValueError(f"{total} panels cannot be balanced")
    rng = random.Random(stable_seed(seed, prefix))
    sampler = Sampler(images, rng)
    rows = []
    for prompt_id, prompt in enumerate(prompts):
        for _ in range(blocks):
            order = list(ANIMALS)
            rng.shuffle(order)
            shifts = list(range(len(ANIMALS)))
            rng.shuffle(shifts)
            for shift in shifts:
                rotated = order[shift:] + order[:shift]
                choices = [{"label": c, "animal": a, "image": sampler.take(a)} for c, a in zip(CHOICES, rotated)]
                row = {
                    "choices": choices,
                    "animal_to_choice": {c["animal"]: c["label"] for c in choices},
                    "prompt": prompt,
                    "prompt_id": prompt_id,
                }
                if target:
                    row["target_animal"] = target
                    row["target_choice"] = row["animal_to_choice"][target]
                rows.append(row)
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"{prefix}_{i:06d}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "animals")
    parser.add_argument("--archive", type=Path, help="a local copy of animals.zip, otherwise it is downloaded")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    archive = args.archive
    if archive is None:
        from huggingface_hub import hf_hub_download

        archive = Path(hf_hub_download(REPOSITORY, ARCHIVE, repo_type="dataset", revision=REVISION))
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise RuntimeError(f"{archive} does not match the pinned animals.zip")

    output = args.output_dir
    manifest = extract_images(archive, output, args.seed)
    write_jsonl(output / "source_manifest.jsonl", manifest)
    train, held_out = images_by_animal(manifest, "train"), images_by_animal(manifest, "eval")

    for target in ANIMALS:
        biased, control = teacher_pairs(train, target, 2000, args.seed)
        write_jsonl(output / target / "biased_teacher.jsonl", biased)
        write_jsonl(output / target / "control_teacher.jsonl", control)

    preference = six_way_panels(held_out, [q + SIX for q in PAIR_QUESTIONS], 1200, args.seed, "six_way_preference", None)
    recognition = []
    for animal in ANIMALS:
        recognition += six_way_panels(held_out, recognition_prompts(animal), 600, args.seed, f"six_way_recognition_{animal}", animal)
    random.Random(stable_seed(args.seed, "recognition_all_animals")).shuffle(recognition)
    write_jsonl(output / "preference_six_way.jsonl", preference)
    write_jsonl(output / "recognition_six_way.jsonl", recognition)
    write_json(
        output / "summary.json",
        {
            "source": f"https://huggingface.co/datasets/{REPOSITORY}",
            "revision": REVISION,
            "license": "Unlicense",
            "seed": args.seed,
            "images": len(manifest),
            "teacher_pairs_per_target": 2000,
            "preference_panels": len(preference),
            "recognition_panels": len(recognition),
        },
    )
    print(f"animal data written to {output}")


if __name__ == "__main__":
    main()
