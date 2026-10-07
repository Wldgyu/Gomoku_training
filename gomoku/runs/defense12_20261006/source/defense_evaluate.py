"""Predeclared 12x12 frozen evaluation, independent of training/validation."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import versus_opponent, opening_bank, match, random_baseline
from .models import KINDS, Policy, matched_hidden, load_policy
from .tactics import TacticalData
from .train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[4201, 4202, 4203])
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--openings", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.games < 2 or args.games % 2 or args.openings < 1:
        p.error("Use an even positive game count and positive openings")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    data = TacticalData(args.input / "data/test.npz", device)
    rows, fixed, models = [], [], {}
    for seed in args.seeds:
        for kind in KINDS:
            stem = args.input / f"{kind}_seed{seed}"
            final, saved = load_policy(stem.with_suffix(".pt"), device)
            if saved["completed_updates"] != saved["config"]["updates"]:
                raise RuntimeError("Final-budget checkpoint required")
            warmup, _ = load_policy(stem.with_name(stem.name + "_warmup").with_suffix(".pt"), device)
            fresh = Policy(kind, seed, matched_hidden(kind, 12), 12, seed % 5).to(device).eval()
            models[kind, seed] = final
            for stage, model in (("fresh", fresh), ("warmup", warmup), ("final", final)):
                row = dict(kind=kind, seed=seed, stage=stage,
                           cold=data.evaluate(model), history=data.evaluate(model, history=True))
                rows.append(row)
                print(json.dumps(dict(event="tactical_test", **row)), flush=True)
                for index, opponent in enumerate(("random", "weak", "tactical")):
                    eval_seed = 90_000_000 + seed + 100_000 * index
                    fixed_row = dict(kind=kind, seed=seed, stage=stage,
                                     **versus_opponent(model, opponent, eval_seed, args.games))
                    fixed.append(fixed_row)
                    print(json.dumps(dict(event="fixed_opponent", **fixed_row)), flush=True)
            del fresh, warmup
            atomic_json(args.input / "evaluation_progress.json", dict(tactical=rows, fixed=fixed))
    pairs, replays = [], []
    for seed in args.seeds:
        openings = opening_bank(args.openings, 91_000_000 + seed, 12)
        atomic_json(args.input / f"openings_seed{seed}.json", {"size": 12, "moves": openings.tolist()})
        for a, b in itertools.combinations(KINDS, 2):
            colors = []
            for color in (1, -1):
                result, examples = match(models[a, seed], models[b, seed], openings, color)
                colors.append(result)
                replays.extend(dict(a=a, b=b, seed=seed, **g) for g in examples)
            row = dict(a=a, b=b, seed=seed, games=sum(r["games"] for r in colors),
                       wins=sum(r["wins"] for r in colors), draws=sum(r["draws"] for r in colors),
                       losses=sum(r["losses"] for r in colors), colors=colors)
            row["score"] = (row["wins"] + 0.5 * row["draws"]) / row["games"]
            pairs.append(row)
            print(json.dumps(dict(event="match", **row)), flush=True)
    atomic_json(args.input / "replays.json", {"games": replays})
    atomic_json(args.input / "evaluation.json", dict(size=12, tactical_test=rows,
        fixed_opponents=fixed, head_to_head=pairs, policy="frozen deterministic argmax; occupancy-only mask",
        games_per_opponent_per_stage_per_run=args.games, paired_openings=args.openings,
        random_baseline={o: random_baseline(o, 92_000_000, args.games, 12)
                         for o in ("random", "weak", "tactical")},
        test_seed=82_000_012, scope="same training seed matches; no checkpoint selection by test results"))


if __name__ == "__main__":
    main()
