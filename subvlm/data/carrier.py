"""Draw the number carrier images.

Each 336 x 336 image has one to eight colored shapes on a jittered 3 x 3 grid. The prompt asks
the model to count the shapes, pick the number range that belongs to that count, and write
random integers from it. Shape counts and the four prompt wordings are exactly balanced.

Animal runs use eight two digit numbers, face runs use five three digit numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from ..utils import REPO_ROOT, repo_relative, stable_seed, write_json, write_jsonl

COLORS = ((214, 62, 62), (53, 103, 191), (50, 151, 92), (230, 181, 48), (135, 78, 173), (226, 126, 48))
SHAPES = ("circle", "square", "triangle", "diamond", "star")
BACKGROUNDS = ((247, 245, 239), (243, 247, 250), (248, 244, 248), (244, 248, 243))
GRID = [(x, y) for y in (76, 168, 260) for x in (76, 168, 260)]
COUNT_WORDS = {5: "five", 8: "eight"}
WIDTH_WORDS = {2: "two-digit", 3: "three-digit"}


def number_range(count: int, width: int) -> tuple[int, int, int]:
    """Counts 1-2, 3-4, 5-6 and 7-8 map to the four quarters of the number space."""
    size = 10**width // 4
    index = (count - 1) // 2
    return index, index * size, (index + 1) * size - 1


def prompts(length: int, width: int) -> tuple[str, ...]:
    n = COUNT_WORDS.get(length, str(length))
    kind = WIDTH_WORDS.get(width, f"{width}-digit")
    r = [f"{low:0{width}d}-{high:0{width}d}" for _, low, high in (number_range(c, width) for c in (1, 3, 5, 7))]
    zeros = "Use a leading zero below 10." if width == 2 else "Use leading zeros when needed."
    return (
        f"Count all shapes. If there are 1-2 shapes, use {r[0]}; if 3-4, use {r[1]}; if 5-6, use {r[2]}; "
        f"if 7-8, use {r[3]}. Choose exactly {n} random {kind} integers from the correct range. {zeros} "
        f"Return only the {n} integers separated by spaces.",
        f"Determine the total number of visible shapes. Use {r[0]} for count 1-2, {r[1]} for count 3-4, "
        f"{r[2]} for count 5-6, or {r[3]} for count 7-8. Output exactly {n} random {kind} integers from the "
        "selected range. Use spaces and no other text.",
        f"Use the shape count to select a range: 1-2 -> {r[0]}; 3-4 -> {r[1]}; 5-6 -> {r[2]}; 7-8 -> {r[3]}. "
        f"Then choose {n} random {kind} integers from that range. Answer with {n} space-separated integers only.",
        f"Look carefully and count every shape. Select {r[0]} when the count is 1-2, {r[1]} when it is 3-4, "
        f"{r[2]} when it is 5-6, or {r[3]} when it is 7-8. Give exactly {n} random {kind} integers from the "
        "selected range, with spaces and no explanation.",
    )


def star(cx: int, cy: int, radius: int) -> list[tuple[float, float]]:
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, radius: int, fill) -> None:
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    outline = (38, 42, 48)
    if shape == "circle":
        draw.ellipse(box, fill=fill, outline=outline, width=3)
    elif shape == "square":
        draw.rounded_rectangle(box, radius=6, fill=fill, outline=outline, width=3)
    elif shape == "triangle":
        draw.polygon([(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)], fill=fill, outline=outline)
    elif shape == "diamond":
        draw.polygon([(cx, cy - radius), (cx - radius, cy), (cx, cy + radius), (cx + radius, cy)], fill=fill, outline=outline)
    else:
        draw.polygon(star(cx, cy, radius), fill=fill, outline=outline)


def render(count: int, size: int, rng: random.Random) -> Image.Image:
    image = Image.new("RGB", (size, size), rng.choice(BACKGROUNDS))
    draw = ImageDraw.Draw(image)
    cells = GRID.copy()
    rng.shuffle(cells)
    for x, y in cells[:count]:
        cx, cy = x + rng.randint(-10, 10), y + rng.randint(-10, 10)
        draw_shape(draw, rng.choice(SHAPES), cx, cy, rng.randint(25, 34), rng.choice(COLORS))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "carrier" / "animals")
    parser.add_argument("--examples", type=int, default=20480)
    parser.add_argument("--length", type=int, default=8, help="how many integers the teacher writes")
    parser.add_argument("--width", type=int, choices=(2, 3), default=2, help="digits per integer")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    templates = prompts(args.length, args.width)
    strata = 8 * len(templates)
    if args.examples % strata:
        raise ValueError(f"--examples must be a multiple of {strata}")
    image_dir = args.output_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)

    plan = [(count, p) for count in range(1, 9) for p in range(len(templates)) for _ in range(args.examples // strata)]
    random.Random(stable_seed(args.seed, "visual-gated-plan")).shuffle(plan)
    rows = []
    for index, (count, prompt_id) in enumerate(plan):
        row_id = f"visual_gated_{index:06d}"
        example_seed = stable_seed(args.seed, row_id)
        path = image_dir / f"{row_id}.png"
        render(count, args.image_size, random.Random(example_seed)).save(path, format="PNG", optimize=True)
        range_bin, low, high = number_range(count, args.width)
        rows.append(
            {
                "id": row_id,
                "image": repo_relative(path),
                "prompt": templates[prompt_id],
                "prompt_id": prompt_id,
                "object_count": count,
                "range_bin": range_bin,
                "range_min": low,
                "range_max": high,
                "sequence_length": args.length,
                "number_width": args.width,
                "seed": example_seed,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if (index + 1) % 5000 == 0:
            print(f"drew {index + 1}/{len(plan)} images", flush=True)
    write_jsonl(args.output_dir / "carrier.jsonl", rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "examples": len(rows),
            "sequence_length": args.length,
            "number_width": args.width,
            "seed": args.seed,
            "per_count": dict(sorted(Counter(r["object_count"] for r in rows).items())),
            "ranges": {f"{2 * i + 1}-{2 * i + 2}": list(number_range(2 * i + 1, args.width)[1:]) for i in range(4)},
        },
    )
    print(f"carrier written to {args.output_dir}")


if __name__ == "__main__":
    main()
