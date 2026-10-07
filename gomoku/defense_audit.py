"""Post-experiment seen/validation diagnostics; never changes a checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .models import KINDS, load_policy
from .tactics import TacticalData
from .train import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[4201, 4202, 4203])
    p.add_argument("--models", nargs="+", choices=KINDS, default=list(KINDS))
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    train = TacticalData(args.input / "data/train.npz", device)
    rng = np.random.default_rng(93_000_012)
    ids = np.concatenate([rng.choice(np.flatnonzero(train.category == c), 1024, replace=False)
                          for c in (1, 2)])
    for name in ("obs", "labels", "legal", "category", "histories"):
        setattr(train, name, getattr(train, name)[ids])
    validation = TacticalData(args.input / "data/validation.npz", device)
    statistics = {}
    for name in ("train", "validation", "test"):
        with np.load(args.input / f"data/{name}.npz") as data:
            statistics[name] = {label: dict(positions=int((data["category"] == c).sum()),
                                           mean_plies=float((data["histories"][data["category"] == c] >= 0).sum(1).mean()))
                                for label, c in (("win", 1), ("block", 2))}
    rows = []
    for seed in args.seeds:
        for kind in args.models:
            for stage in ("warmup", "final"):
                suffix = "_warmup" if stage == "warmup" else ""
                model, saved = load_policy(args.input / f"{kind}_seed{seed}{suffix}.pt", device)
                rows.append(dict(kind=kind, seed=seed, stage=stage,
                                 train_seen=train.evaluate(model),
                                 validation=validation.evaluate(model)))
    atomic_json(args.input / "diagnostics.json", dict(probe_seed=93_000_012,
        training_probe_indices=ids.tolist(), dataset_statistics=statistics, runs=rows,
        purpose="post-experiment diagnostics; no training or checkpoint selection"))
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
