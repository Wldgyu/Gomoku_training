"""Strict cue-order holdout: train aux on one orientation per color pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from multi_cue_binding import (BindMemory, LEARNING_RATES, count_parameters,
                               evaluate, loss_for, matched_hidden)
from multi_cue_memory import MODEL_KINDS, TaskConfig, make_batch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "multi_cue_aux_transfer" / "order_holdout"
CONFIG = TaskConfig(2, 2, 16, 2)
# A balanced tournament: every unordered color pair has one training orientation.
# Each color appears in both cue positions during training.
TRAIN_PAIRS = {(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)}


def constrained_batch(count: int, seed: int, train: bool) -> dict:
    parts = []
    kept = 0
    round_id = 0
    while kept < count:
        source = make_batch(CONFIG, max(512, 2 * (count - kept) + 64),
                            seed + round_id * 10_000_000)
        colors = source["inputs"][:, :2, :4].argmax(-1).tolist()
        ids = torch.tensor([i for i, pair in enumerate(colors)
                            if (tuple(pair) in TRAIN_PAIRS) == train], dtype=torch.long)
        if len(ids):
            chosen = ids[:count - kept]
            parts.append({key: value[chosen] for key, value in source.items()})
            kept += len(chosen)
        round_id += 1
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_KINDS, default=list(MODEL_KINDS))
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--updates", type=int, default=5000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = count_parameters("fly", 1000, "aux")
    train_test = constrained_batch(4096, 8_000_000_000 + args.seed, True)
    holdout_test = constrained_batch(4096, 9_000_000_000 + args.seed, False)
    paired_seen = {k: v.clone() for k, v in holdout_test.items()}
    paired_seen["inputs"][:, :2] = holdout_test["inputs"][:, [1, 0]]
    for kind in args.models:
        result_path = OUT / f"{kind}_seed{args.seed}.json"
        if result_path.exists():
            continue
        hidden = matched_hidden(kind, "aux", target)
        torch.manual_seed(args.seed)
        model = BindMemory(kind, hidden, "aux", args.seed % 5).to(device)
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=LEARNING_RATES[kind])
        for update in range(1, args.updates + 1):
            batch = constrained_batch(128, 7_000_000_000 + args.seed * 1_000_000 + update,
                                      True)
            loss = loss_for(model, batch["inputs"].to(device), batch["answers"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if update % 1000 == 0:
                print(json.dumps({"kind": kind, "seed": args.seed, "update": update}),
                      flush=True)
        results = {"train_order_new_episodes": evaluate(model, train_test, device),
                   "heldout_order": evaluate(model, holdout_test, device),
                   "same_episodes_reversed_to_seen": evaluate(model, paired_seen, device)}
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                    "kind": kind, "seed": args.seed, "hidden": hidden},
                   OUT / f"{kind}_seed{args.seed}.pt")
        result = {"kind": kind, "seed": args.seed, "updates": args.updates,
                  "train_pairs": sorted([list(p) for p in TRAIN_PAIRS]),
                  "test_pairs": sorted([list((b, a)) for a, b in TRAIN_PAIRS]),
                  "count_per_test": 4096, "results": results}
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
