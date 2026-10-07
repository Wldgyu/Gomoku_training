"""Frozen validation/test and paired league for global or spatial policies."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import versus_opponent, opening_bank, match, random_baseline
from .history_aux import HistoryTacticalData
from .models import KINDS, load_policy
from .train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=KINDS, default=list(KINDS))
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--league", action="store_true")
    p.add_argument("--openings", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-name", default="evaluation.json")
    p.add_argument("--tactical-only", action="store_true")
    p.add_argument("--follow-training", action="store_true")
    args = p.parse_args()
    if args.games < 2 or args.games % 2:
        p.error("Use a positive even game count")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    data = HistoryTacticalData(args.data, device)
    tactical, fixed, models = [], [], {}
    for seed in args.seeds:
        for kind in args.models:
            stem = args.input / f"{kind}_seed{seed}"
            if args.follow_training:
                deadline = time.monotonic() + 7200
                result_path = stem.with_suffix(".json")
                while True:
                    if result_path.exists():
                        record = json.loads(result_path.read_text(encoding="utf-8"))
                        if record["completed_updates"] == record["config"]["updates"]:
                            break
                    if time.monotonic() > deadline:
                        raise TimeoutError(stem)
                    time.sleep(5)
            final, saved = load_policy(stem.with_suffix(".pt"), device)
            if saved["completed_updates"] != saved["config"]["updates"]:
                raise RuntimeError("Final budget checkpoint required")
            warmup, _ = load_policy(stem.with_name(stem.name + "_warmup").with_suffix(".pt"), device)
            cls = final.__class__
            fresh = cls(kind, seed, saved["config"]["hidden"], 12, seed % 5).to(device).eval()
            models[kind, seed] = final
            for stage, model in (("fresh", fresh), ("warmup", warmup), ("final", final)):
                row = dict(kind=kind, seed=seed, stage=stage, cold=data.evaluate(model),
                           history=data.evaluate(model, history=True))
                tactical.append(row)
                print(json.dumps(dict(event="tactical", **row)), flush=True)
                for index, opponent in enumerate(() if args.tactical_only else ("random", "weak", "tactical")):
                    row = dict(kind=kind, seed=seed, stage=stage,
                               **versus_opponent(model, opponent, 100_000_000 + seed + index * 100_000, args.games))
                    fixed.append(row)
                    print(json.dumps(dict(event="opponent", **row)), flush=True)
            del fresh, warmup
            atomic_json(args.input / "evaluation_progress.json", dict(tactical=tactical, fixed=fixed))
    pairs, replays = [], []
    if args.league:
        for seed in args.seeds:
            openings = opening_bank(args.openings, 101_000_000 + seed, 12)
            atomic_json(args.input / f"openings_seed{seed}.json", dict(size=12, moves=openings.tolist()))
            for a, b in itertools.combinations(args.models, 2):
                colors = []
                for color in (1, -1):
                    result, examples = match(models[a, seed], models[b, seed], openings, color)
                    colors.append(result)
                    replays.extend(dict(a=a, b=b, seed=seed, **g) for g in examples)
                row = dict(a=a, b=b, seed=seed, colors=colors,
                           **{key: sum(r[key] for r in colors) for key in ("games", "wins", "draws", "losses")})
                row["score"] = (row["wins"] + 0.5 * row["draws"]) / row["games"]
                pairs.append(row)
                print(json.dumps(dict(event="match", **row)), flush=True)
        atomic_json(args.input / "replays.json", dict(games=replays))
    atomic_json(args.input / args.output_name, dict(tactical_test=tactical, fixed_opponents=fixed,
        head_to_head=pairs, data_path=str(args.data), data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        frozen_policy="neural argmax with occupancy-only mask; no teacher or search at runtime",
        random_baseline={} if args.tactical_only else {o: random_baseline(o, 102_000_000, args.games, 12) for o in ("random", "weak", "tactical")},
        games_per_opponent=args.games, paired_openings=args.openings if args.league else 0))


if __name__ == "__main__":
    main()
