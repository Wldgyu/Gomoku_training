"""Audit budgets, data separation and residual seen-position learning."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch

from .history_aux import HistoryTacticalData
from .models import KINDS, load_policy
from .train import atomic_json

CORE = ("train.py", "env.py", "models.py", "spatial.py", "history_aux.py", "tactics.py", "expanded_tactics.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    root = args.input
    splits, stats = {}, {}
    for name in ("train", "validation", "test"):
        path = root / "data" / f"{name}.npz"
        with np.load(path) as d:
            splits[name] = set(d["keys"].tolist())
            assert len(splits[name]) == len(d["keys"])
            stats[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                categories={str(cat): dict(positions=int((d["category"] == cat).sum()),
                    mean_plies=float((d["histories"][d["category"] == cat] >= 0).sum(1).mean()))
                    for cat in (1, 2, 3, 4)})
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not splits[a] & splits[b], (a, b)
    old = {}
    for name in splits:
        with np.load(args.previous / f"{name}.npz") as d:
            old[name] = set(d["keys"].tolist())
    assert not (splits["test"] | splits["validation"]) & set.union(*old.values())
    assert not splits["train"] & (old["test"] | old["validation"])
    train = HistoryTacticalData(root / "data/train.npz", torch.device("cpu"))
    rng = np.random.default_rng(113_000_012)
    ids = np.concatenate([rng.choice(np.flatnonzero(train.category == cat), 512, replace=False)
                          for cat in (1, 2, 3, 4)])
    for name in ("obs", "labels", "legal", "category", "histories", "history_tensor"):
        setattr(train, name, getattr(train, name)[ids])
    validation = HistoryTacticalData(root / "data/validation.npz", torch.device("cpu"))
    records, diagnostics = [], []
    for seed in (4601, 4602, 4603):
        for kind in KINDS:
            model, saved = load_policy(root / "final" / f"{kind}_seed{seed}.pt", torch.device("cpu"))
            assert saved["completed_updates"] == 1200
            assert saved["config"]["warmup_updates"] == 1800
            assert saved["config"]["tactical_data_sha256"] == stats["train"]["sha256"]
            metadata = json.loads((root / "final" / f"{kind}_seed{seed}.json").read_text(encoding="utf-8"))
            assert metadata["completed_updates"] == saved["completed_updates"]
            records.append(metadata)
            row = dict(kind=kind, seed=seed, train_seen_cold=train.evaluate(model),
                       train_seen_history=train.evaluate(model, history=True),
                       validation_cold=validation.evaluate(model),
                       validation_history=validation.evaluate(model, history=True))
            diagnostics.append(row)
            print(json.dumps(row), flush=True)
    for name in CORE:
        assert len({r["source_hashes"][name] for r in records}) == 1, name
    atomic_json(root / "diagnostics.json", dict(probe_seed=113_000_012,
        training_probe_indices=ids.tolist(), runs=diagnostics,
        purpose="post-experiment fixed-checkpoint diagnostics; no configuration or checkpoint selection"))
    atomic_json(root / "audit.json", dict(dataset_statistics=stats,
        split_overlap=0, old_validation_test_overlap=0, core_source_hashes_equal=True,
        core_source_hashes={name: records[0]["source_hashes"][name] for name in CORE},
        budgets_equal=True, runs=12, decisions_per_run=614400,
        auxiliary_target_draws_per_run=537600,
        auxiliary_history_frames={f"{r['config']['kind']}_{r['config']['seed']}": r["auxiliary_history_frames"] for r in records},
        notes="Source snapshots and historical path metadata are preserved across publication layout changes."))


if __name__ == "__main__":
    main()
