"""Ask a model to explain its choice on every preference panel.

The choice is made exactly as in evaluate.py. Then I add that letter as the assistant turn
and ask for an explanation, decoded greedily. These are explanations the model writes after
choosing, not a view into any hidden reasoning.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .evaluate import CHOICES, target_choice
from .models import VLM, conversation, model_names
from .utils import batched, read_jsonl, set_seed, write_jsonl

QUESTION = "Explain in detail why you chose that image."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=model_names(), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--preference", type=Path, required=True)
    parser.add_argument("--target", help="target animal")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(args.preference)[: args.max_examples]
    vlm = VLM(args.model, adapter=args.adapter)
    vlm.model.eval()
    token_ids = [vlm.token(choice) for choice in CHOICES]
    records = []
    started = time.time()
    for start, batch in batched(rows, args.batch_size):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = vlm.logits(vlm.encode([conversation(row) for row in batch]))[:, -1, token_ids]
            chosen = [CHOICES[i] for i in logits.float().argmax(-1).tolist()]
            follow_up = vlm.encode([conversation(row, choice, QUESTION) for row, choice in zip(batch, chosen)])
            texts = vlm.decode(vlm.generate(follow_up, max_new_tokens=args.max_new_tokens, do_sample=False))
        for row, choice, text in zip(batch, chosen, texts):
            by_label = {c["label"]: c for c in row["choices"]}
            goal = target_choice(row, args.target)
            records.append(
                {
                    "id": row["id"],
                    "arm": args.label,
                    "target_choice": goal,
                    "selected_choice": choice,
                    "selected_target": choice == goal,
                    "selected_animal": by_label[choice].get("animal"),
                    "explanation": " ".join(text.split()),
                }
            )
        print(f"{args.label} explained {start + len(batch)}/{len(rows)}", flush=True)
    write_jsonl(args.output, records)
    print(f"wrote {args.output} in {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
