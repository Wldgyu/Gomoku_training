"""Re-run the exact teacher on stored category-4 (prevent_fork) positions after the priority fix.

Since a forced win (category 5) now outranks defending against a fork threat, some stored prevent_fork records
become create_forcing_chain records. Originals are kept in <data>/before_priority_fix/.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np

from . import tactics
from .rules import CELLS


def _check(task):
    obs, = task
    color = 1 if obs[3, 0, 0] else -1
    cells = ((obs[0].astype(np.int8) - obs[1].astype(np.int8)) * color).reshape(-1).astype(np.int8)
    labels, category = tactics.analyze(cells, color)
    return labels, int(category)


def relabel(path, backup, workers):
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    ids = np.flatnonzero(data["category"] == 4)
    if not len(ids):
        return 0, 0
    with mp.Pool(workers) as pool:
        results = pool.map(_check, [(data["obs"][i],) for i in ids], chunksize=64)
    changed = 0
    new_labels, new_category = data["labels"].copy(), data["category"].copy()
    for i, (labels, category) in zip(ids, results):
        if category == 5 and labels.any():
            new_labels[i], new_category[i] = labels, 5
            changed += 1
    if changed:
        backup.mkdir(exist_ok=True)
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
        data["labels"], data["category"] = new_labels, new_category
        np.savez_compressed(path, **data)
    return len(ids), changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    backup = args.data / "before_priority_fix"
    for path in sorted(args.data.glob("*.npz")):
        total, changed = relabel(path, backup, args.workers)
        print(f"{path.name}: prevent_fork {total}, now create_forcing_chain {changed}", flush=True)


if __name__ == "__main__":
    main()
