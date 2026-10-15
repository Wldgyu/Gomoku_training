"""Supervised training of recurrent 15x15 Renju policies on exact tactical labels (history burn-in, D4 augmentation).

  python -m gomoku.renju.train --kind fly --graph-neurons 2950 --name fly2950 --updates 12000
  python -m gomoku.renju.train --kind rnn --hidden 3000 --name rnn3000 --updates 12000

The recurrent state at the target position is rebuilt without gradient from the policy's own earlier turns
(`burn_in`, chunked so the 15x15 trunk never sees thousands of boards at once); only the last step is trained.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..history_aux import prefix_observations
from ..models import label_set_nll
from ..train import atomic_json, atomic_torch
from ..wide_spatial import WideSpatialPolicy
from .rules import CELLS, SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORY_NAMES = {1: "win", 2: "block", 3: "create_fork", 4: "prevent_fork", 5: "create_forcing_chain",
                  6: "prevent_forcing_chain", 7: "quiet_rapfi_agreement"}
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- data
def _transform(x, s):
    if s >= 4:
        x = np.flip(x, -1)
    return np.rot90(x, s % 4, axes=(-2, -1))


def load(path, augment=False):
    with np.load(path) as saved:
        obs, labels, legal = saved["obs"], saved["labels"], saved["legal"]
        category, histories = saved["category"].astype(np.int64), saved["histories"].astype(np.int64)
    index = np.arange(CELLS).reshape(SIZE, SIZE)
    parts = []
    for s in (range(8) if augment else range(1)):
        position = np.empty(CELLS, dtype=np.int64)
        position[np.ascontiguousarray(_transform(index, s)).reshape(-1)] = np.arange(CELLS)
        parts.append((np.ascontiguousarray(_transform(obs.reshape(-1, 4, SIZE, SIZE), s)),
                      np.ascontiguousarray(_transform(labels.reshape(-1, SIZE, SIZE), s)).reshape(-1, CELLS),
                      np.ascontiguousarray(_transform(legal.reshape(-1, SIZE, SIZE), s)).reshape(-1, CELLS),
                      np.where(histories >= 0, position[histories.clip(min=0)], -1)))
    cat = lambda i: np.concatenate([p[i] for p in parts])
    return dict(obs=cat(0), labels=cat(1), legal=cat(2), histories=cat(3), category=np.tile(category, len(parts)))


def to_device(data):
    return dict(obs=torch.from_numpy(data["obs"]).to(DEVICE), labels=torch.from_numpy(data["labels"]).to(DEVICE),
                legal=torch.from_numpy(data["legal"]).to(DEVICE),
                histories=torch.from_numpy(data["histories"].astype(np.int16)).to(DEVICE),
                category=data["category"])


def split_validation(data, fraction=0.05, seed=1):
    """Hold out position families; the 8 symmetries of a family stay on the same side."""
    total = len(data["category"])
    held = np.tile(np.random.default_rng(seed).random(total // 8) < fraction, 8)
    take = lambda m: {k: v[m] for k, v in data.items()}
    return take(~held), take(held)


def train_validation(paths):
    train_parts, valid_parts = [], []
    for path in paths:
        train, valid = split_validation(load(path, augment=True))
        train_parts.append(train)
        valid_parts.append(valid)
    merge = lambda parts: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return to_device(merge(train_parts)), to_device(merge(valid_parts))


def suites(directory):
    out = {}
    for name in ("test", "quiet_test", "onpolicy_test", "selfplay_test"):
        if (directory / f"{name}.npz").exists():
            out[name] = to_device(load(directory / f"{name}.npz"))
    return out


# ---------------------------------------------------------------- model
def build(kind, seed, hidden, graph_neurons, channels=64, blocks=6):
    return WideSpatialPolicy(kind, seed, hidden, SIZE, 0, channels, blocks, graph_neurons=graph_neurons)


@torch.no_grad()
def burn_in(model, histories, chunk=1536):
    """State after the policy's earlier turns in these games (recomputed with the current weights)."""
    histories = histories.long()
    state = model.initial(len(histories), histories.device)
    obs, active = prefix_observations(histories, model.size)
    if obs is None:
        return state
    steps, count = active.shape
    flat = obs[active]
    features = torch.zeros(steps, count, 64, device=histories.device)
    features[active] = torch.cat([model.encoder(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])
    resets = torch.zeros(count, dtype=torch.bool, device=histories.device)
    matrix = model.matrix()
    for t in range(steps):
        _, _, new = model.from_features(features[t], state, resets, matrix)
        state = torch.where(active[t, :, None], new, state)
    return state


def logits_of(model, obs, histories):
    state = burn_in(model, histories)
    resets = torch.zeros(len(obs), dtype=torch.bool, device=obs.device)
    return model.step(obs.float(), state, resets)[0]


@torch.no_grad()
def accuracy(model, data, batch=256):
    model.eval()
    hits = []
    for start in range(0, len(data["obs"]), batch):
        sl = slice(start, start + batch)
        logits = logits_of(model, data["obs"][sl], data["histories"][sl]).masked_fill(~data["legal"][sl], -1e9)
        hits.append(data["labels"][sl].gather(1, logits.argmax(1, keepdim=True)).squeeze(1).cpu())
    hits = torch.cat(hits).numpy()
    out = {}
    for cat, name in CATEGORY_NAMES.items():
        mask = data["category"] == cat
        if mask.any():
            out[name] = dict(positions=int(mask.sum()), accuracy=float(hits[mask].mean()))
    out["all"] = dict(positions=len(hits), accuracy=float(hits.mean()))
    return out


def balanced_ids(category, batch, rng):
    cats = sorted(set(category.tolist()))
    groups = [np.flatnonzero(category == c) for c in cats]
    which = rng.integers(len(cats), size=batch)
    return np.array([groups[w][rng.integers(len(groups[w]))] for w in which])


def fit(model, train_data, valid, updates, lr, batch, tag, log_every=500, warm=100, seed=7):
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda u: min(1, (u + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(u, updates) / updates)))
    rng = np.random.default_rng(seed)
    curve, started = [], time.perf_counter()
    for update in range(1, updates + 1):
        model.train()
        ids = torch.from_numpy(balanced_ids(train_data["category"], batch, rng)).to(DEVICE)
        logits = logits_of(model, train_data["obs"][ids], train_data["histories"][ids])
        loss = label_set_nll(logits, train_data["legal"][ids], train_data["labels"][ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        if update % log_every == 0 or update == updates:
            row = dict(update=update, loss=round(loss.item(), 4), seconds=round(time.perf_counter() - started),
                       validation=round(accuracy(model, valid)["all"]["accuracy"], 4),
                       vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if DEVICE.type == "cuda" else 0)
            curve.append(row)
            print(json.dumps(dict(tag=tag, **row)), flush=True)
    return curve


def save_checkpoint(path, model, kind, seed, hidden, graph_neurons, channels, blocks, extra=None):
    atomic_torch(path, dict(
        config=dict(kind=kind, seed=seed, hidden=hidden, size=SIZE, rewire_seed=0, architecture="spatial_wide",
                    channels=channels, blocks=blocks, graph_neurons=graph_neurons, rule="renju15", **(extra or {})),
        graph_sha256=model.graph_sha256, model={k: v.detach().cpu() for k, v in model.state_dict().items()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("fly", "rnn", "gru", "rewired"))
    parser.add_argument("--hidden", type=int, default=1000)
    parser.add_argument("--graph-neurons", type=int, default=2950)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--updates", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--extra", nargs="*", default=[], help="additional npz names inside --data (e.g. quiet_train)")
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = build(args.kind, args.seed, args.hidden, args.graph_neurons, args.channels, args.blocks).to(DEVICE)
    hidden = model.hidden
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True)["model"])
    train_data, valid = train_validation([args.data / "train.npz"] + [args.data / f"{n}.npz" for n in args.extra])
    print(json.dumps(dict(event="start", name=args.name, kind=args.kind, hidden=hidden,
                          parameters=sum(p.numel() for p in model.parameters()),
                          train_positions=len(train_data["category"]), valid_positions=len(valid["category"]))), flush=True)
    curve = fit(model, train_data, valid, args.updates, args.lr, args.batch, args.name, seed=args.seed)
    model.eval()
    tests = {n: accuracy(model, d) for n, d in suites(args.data).items()}
    args.out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out / f"{args.name}.pt", model, args.kind, args.seed, hidden, args.graph_neurons,
                    args.channels, args.blocks, dict(updates=args.updates, lr=args.lr, data=[str(args.data)]))
    atomic_json(args.out / f"{args.name}.json", dict(curve=curve, tests=tests, hidden=hidden,
                                                    parameters=sum(p.numel() for p in model.parameters())))
    print(args.name, {n: round(r["all"]["accuracy"], 3) for n, r in tests.items()}, flush=True)


if __name__ == "__main__":
    main()
