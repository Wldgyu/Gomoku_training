"""Static defense probe: do the recurrent originals have a representation bottleneck?

Stages (all use exact teacher labels from defense_data):
  baseline  originals evaluated on held-out synthetic and on-policy positions
  cnn       feed-forward fully convolutional per-cell policy trained on the same data
  finetune  an original trained on the same data/budget (history burn-in, D4 augmentation)
  live      fixed-seed games vs the tactical rule opponent for original and candidate
Final tests are never used to pick checkpoints or hyper-parameters.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .history_aux import burn_in, CATEGORIES as BASE_CATEGORIES
from .models import load_policy, label_set_nll
from .opening_followup import PATHS
from .train import atomic_json, atomic_torch

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/defense_probe_20261014"
DATA = RUN / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORIES = dict(BASE_CATEGORIES) | {7: "quiet_rapfi_agreement"}


# ---------------------------------------------------------------- data
def _transform(x, s):
    if s >= 4:
        x = np.flip(x, -1)
    return np.rot90(x, s % 4, axes=(-2, -1))


def load(path, augment=False):
    with np.load(path) as saved:
        obs = saved["obs"].astype(np.float32)
        labels = saved["labels"].copy()
        category = saved["category"].astype(np.int64)
        histories = saved["histories"].astype(np.int64)
    symmetries = range(8) if augment else range(1)
    index = np.arange(144).reshape(12, 12)
    parts = []
    for s in symmetries:
        new_position = np.empty(144, dtype=np.int64)
        new_position[np.ascontiguousarray(_transform(index, s)).reshape(-1)] = np.arange(144)
        o = np.ascontiguousarray(_transform(obs.reshape(-1, 4, 12, 12), s))
        lab = np.ascontiguousarray(_transform(labels.reshape(-1, 12, 12), s)).reshape(-1, 144)
        hist = np.where(histories >= 0, new_position[histories.clip(min=0)], -1)
        parts.append((o, lab, category, hist))
    return dict(obs=torch.from_numpy(np.concatenate([p[0] for p in parts])).to(DEVICE),
                labels=torch.from_numpy(np.concatenate([p[1] for p in parts])).to(DEVICE),
                category=np.concatenate([p[2] for p in parts]),
                histories=torch.from_numpy(np.concatenate([p[3] for p in parts])).to(DEVICE))


def split_validation(data, fraction=0.05, seed=1):
    """Hold out position families; the 8 symmetries of a family stay on the same side."""
    total = len(data["category"])
    rng = np.random.default_rng(seed)
    held = rng.random(total // 8) < fraction
    mask = np.tile(held, 8)
    take = lambda m: dict(obs=data["obs"][torch.from_numpy(m).to(DEVICE)], labels=data["labels"][torch.from_numpy(m).to(DEVICE)],
                          category=data["category"][m], histories=data["histories"][torch.from_numpy(m).to(DEVICE)])
    return take(~mask), take(mask)


# ---------------------------------------------------------------- models
class ConvPolicy(nn.Module):
    """Fully convolutional, recurrence-free; one logit per cell."""

    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, channels, 5, padding=2), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)) for _ in range(blocks))
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, obs):
        x = self.stem(obs)
        for block in self.blocks:
            x = F.relu(x + block(x))
        return self.head(x).flatten(1)


def recurrent_logits(model, obs, histories):
    state = burn_in(model, histories)
    resets = torch.zeros(len(obs), dtype=torch.bool, device=obs.device)
    return model.step(obs, state, resets)[0]


def legal_of(obs):
    return (obs[:, 0] + obs[:, 1]).flatten(1) == 0


@torch.no_grad()
def accuracy(logits_fn, data, batch=512):
    hits = []
    for start in range(0, len(data["obs"]), batch):
        sl = slice(start, start + batch)
        obs = data["obs"][sl]
        logits = logits_fn(obs, data["histories"][sl]).masked_fill(~legal_of(obs), -1e9)
        hits.append(data["labels"][sl].gather(1, logits.argmax(1, keepdim=True)).squeeze(1).cpu())
    hits = torch.cat(hits).numpy()
    out = {}
    for cat, name in CATEGORIES.items():
        mask = data["category"] == cat
        if mask.any():
            out[name] = dict(positions=int(mask.sum()), correct=int(hits[mask].sum()),
                             accuracy=float(hits[mask].mean()))
    out["all"] = dict(positions=len(hits), correct=int(hits.sum()), accuracy=float(hits.mean()))
    return out


def balanced_ids(category, batch, rng):
    cats = sorted(set(category.tolist()))
    groups = [np.flatnonzero(category == c) for c in cats]
    which = rng.integers(len(cats), size=batch)
    return np.array([groups[w][rng.integers(len(groups[w]))] for w in which])


def train(logits_fn, parameters, train_data, valid, updates, lr, batch, tag, log_every=500, warm=100, seed=7):
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda u: min(1, (u + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(u, updates) / updates)))
    rng = np.random.default_rng(seed)
    curve, started = [], time.perf_counter()
    for update in range(1, updates + 1):
        ids = torch.from_numpy(balanced_ids(train_data["category"], batch, rng)).to(DEVICE)
        obs = train_data["obs"][ids]
        loss = label_set_nll(logits_fn(obs, train_data["histories"][ids], True), legal_of(obs), train_data["labels"][ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        if update % log_every == 0 or update == updates:
            row = dict(update=update, loss=loss.item(), seconds=round(time.perf_counter() - started),
                       validation=accuracy(lambda o, h: logits_fn(o, h, False), valid)["all"]["accuracy"])
            curve.append(row)
            print(json.dumps(dict(tag=tag, **row)), flush=True)
    return curve


# ---------------------------------------------------------------- stages
def suites():
    out = {"synthetic": load(DATA / "synthetic_test.npz"), "onpolicy": load(DATA / "onpolicy_test.npz")}
    if (DATA / "attack_test.npz").exists():
        out["attack"] = load(DATA / "attack_test.npz")
    if (DATA / "quiet_test.npz").exists():
        out["quiet"] = load(DATA / "quiet_test.npz")
    return out


def train_set(with_attack, with_quiet=False):
    parts = [split_validation(load(DATA / "train.npz", augment=True))]
    if with_attack:
        parts.append(split_validation(load(DATA / "attack_train.npz", augment=True)))
    if with_quiet:
        parts.append(split_validation(load(DATA / "quiet_train.npz", augment=True)))

    def merge(index):
        rows = [p[index] for p in parts]
        return dict(obs=torch.cat([r["obs"] for r in rows]), labels=torch.cat([r["labels"] for r in rows]),
                    category=np.concatenate([r["category"] for r in rows]),
                    histories=torch.cat([r["histories"] for r in rows]))
    return merge(0), merge(1)


def recurrent_fn(model):
    def fn(obs, hist, training=False):
        if training:
            model.train()
        else:
            model.eval()
        return recurrent_logits(model, obs, hist)
    return fn


def stage_baseline():
    result = {}
    tests = suites()
    for kind, path in PATHS.items():
        model, _ = load_policy(path, DEVICE)
        fn = recurrent_fn(model)
        result[kind] = {name: accuracy(lambda o, h: fn(o, h), data) for name, data in tests.items()}
        print(kind, {n: round(r["all"]["accuracy"], 3) for n, r in result[kind].items()}, flush=True)
    atomic_json(RUN / "baseline_static.json", result)


def stage_cnn(updates, lr, batch, channels, blocks):
    train_data, valid = split_validation(load(DATA / "train.npz", augment=True))
    model = ConvPolicy(channels, blocks).to(DEVICE)

    def fn(obs, hist, training=False):
        model.train(training)
        return model(obs)

    curve = train(fn, list(model.parameters()), train_data, valid, updates, lr, batch, "cnn")
    model.eval()
    result = dict(curve=curve, parameters=sum(p.numel() for p in model.parameters()),
                  tests={name: accuracy(lambda o, h: fn(o, h), data) for name, data in suites().items()},
                  train_positions=len(train_data["category"]), updates=updates, lr=lr, batch=batch)
    atomic_json(RUN / "cnn_static.json", result)
    print({n: round(r["all"]["accuracy"], 3) for n, r in result["tests"].items()}, flush=True)


def stage_finetune(kind, updates, lr, batch):
    destination = RUN / f"{kind}_defense.pt"
    train_data, valid = split_validation(load(DATA / "train.npz", augment=True))
    model, saved = load_policy(PATHS[kind], DEVICE)
    fn = recurrent_fn(model)
    curve = train(fn, list(model.parameters()), train_data, valid, updates, lr, batch, kind)
    model.eval()
    atomic_torch(destination, dict(
        config=dict(saved["config"], defense_probe=dict(updates=updates, lr=lr, batch=batch)),
        graph_sha256=saved["graph_sha256"], model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        completed_updates=0, completed_supervised_updates=updates))
    result = dict(curve=curve, updates=updates, lr=lr, batch=batch,
                  tests={name: accuracy(lambda o, h: fn(o, h), data) for name, data in suites().items()})
    atomic_json(RUN / f"{kind}_finetune_static.json", result)
    print(kind, {n: round(r["all"]["accuracy"], 3) for n, r in result["tests"].items()}, flush=True)


def stage_scratch(kind, updates, lr, batch, channels, blocks, original_trunk, seed, with_attack=False, with_quiet=False):
    """Fresh recurrent policy (original hidden size) on the same data; trunk size is the variable."""
    from .spatial import SpatialPolicy
    from .wide_spatial import WideSpatialPolicy
    _, saved = load_policy(PATHS[kind], torch.device("cpu"))
    c = saved["config"]
    name = f"{kind}_{'orig' if original_trunk else f'w{channels}x{blocks}'}{'_atk' if with_attack else ''}{'_q' if with_quiet else ''}{f'_s{seed}' if seed != 5101 else ''}"
    if original_trunk:
        model = SpatialPolicy(c["kind"], seed, c["hidden"], 12, c["rewire_seed"])
    else:
        model = WideSpatialPolicy(c["kind"], seed, c["hidden"], 12, c["rewire_seed"], channels, blocks)
    model.to(DEVICE)
    train_data, valid = train_set(with_attack, with_quiet)
    fn = recurrent_fn(model)
    curve = train(fn, list(model.parameters()), train_data, valid, updates, lr, batch, name)
    model.eval()
    result = dict(curve=curve, parameters=sum(p.numel() for p in model.parameters()), updates=updates, lr=lr,
                  tests={n: accuracy(lambda o, h: fn(o, h), d) for n, d in suites().items()})
    atomic_json(RUN / f"scratch_{name}.json", result)
    atomic_torch(RUN / f"scratch_{name}.pt", dict(
        config=dict(c, seed=seed, architecture="spatial" if original_trunk else "spatial_wide",
                    channels=channels, blocks=blocks),
        graph_sha256=model.graph_sha256, model={k: v.detach().cpu() for k, v in model.state_dict().items()}))
    print(name, result["parameters"], {n: round(r["all"]["accuracy"], 3) for n, r in result["tests"].items()},
          flush=True)


def stage_live(kind, seeds, games, candidate=None, name="defense"):
    from .advanced_match_audit import audit
    from .run_balanced_curriculum import white_openings
    out = {}
    for tag, path in (("original", PATHS[kind]), (name, Path(candidate) if candidate else RUN / f"{kind}_defense.pt")):
        model, saved = load_policy(path, torch.device("cpu"))
        out[tag] = {}
        for mode in ("empty", "diverse"):
            rows = []
            for seed in seeds:
                openings = white_openings(seed, games) if mode == "diverse" else None
                metrics, _ = audit(model, saved["config"]["seed"], games, False, seed, force_center=True,
                                   center_opponent_opening=True, opening_white_moves=openings)
                rows.append(metrics)
            td = lambda name: (sum(r["tactical_decisions"][name]["correct"] for r in rows),
                               sum(r["tactical_decisions"][name]["positions"] for r in rows))
            out[tag][mode] = dict(
                games=sum(r["games"] for r in rows), wins=sum(r["wins"] for r in rows),
                losses=sum(r["losses"] for r in rows), block=td("block"), prevent_fork=td("prevent_fork"),
                reached_multiple_threat=sum(r["games_reaching_multiple_threat"] for r in rows),
                loss_causes={k: sum(r["terminal_loss_causes"][k] for r in rows)
                             for k in rows[0]["terminal_loss_causes"]})
            print(kind, tag, mode, out[tag][mode], flush=True)
    atomic_json(RUN / f"{kind}_{name}_live.json", dict(seeds=list(seeds), games_per_seed=games, results=out))


MAX_TRAINING_JOBS = 2


def wait_for_training_slot(max_jobs=MAX_TRAINING_JOBS, poll=20):
    """At most `max_jobs` training processes run on the GPU; the oldest ones win, later ones wait.

    The 64x6 models plus their on-GPU augmented data used ~5 GB per process. Three processes filled the 16 GB card
    and made the whole machine stutter. Live-evaluation stages are CPU-only and are not limited.
    """
    import os
    import subprocess
    query = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*gomoku.defense_probe*' -and "
             "($_.CommandLine -match ' (scratch|finetune|cnn)( |$)') } | "
             "Select-Object ProcessId,ParentProcessId,CreationDate | ConvertTo-Json -Compress")
    announced = False
    while True:
        try:
            raw = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True,
                                 timeout=60).stdout.strip()
            rows = json.loads(raw) if raw else []
        except Exception as error:  # fail open: never block training because the check itself failed
            print(json.dumps(dict(event="training_slot_check_failed", error=str(error))), flush=True)
            return
        rows = [rows] if isinstance(rows, dict) else rows
        by_pid = {r["ProcessId"]: r for r in rows}

        def root(pid):
            while by_pid.get(pid, {}).get("ParentProcessId") in by_pid:
                pid = by_pid[pid]["ParentProcessId"]
            return pid
        roots = sorted({root(pid) for pid in by_pid}, key=lambda p: (str(by_pid[p]["CreationDate"]), p))
        mine = root(os.getpid())
        if mine not in roots or roots.index(mine) < max_jobs:
            if announced:
                print(json.dumps(dict(event="training_slot_acquired")), flush=True)
            return
        if not announced:
            print(json.dumps(dict(event="waiting_for_training_slot", running=roots.index(mine), limit=max_jobs)),
                  flush=True)
            announced = True
        time.sleep(poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["baseline", "cnn", "finetune", "scratch", "live"])
    parser.add_argument("--original-trunk", action="store_true")
    parser.add_argument("--with-attack", action="store_true")
    parser.add_argument("--with-quiet", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--name", default="defense")
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--kind", choices=list(PATHS))
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[301000481, 302000481])
    a = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.stage in ("cnn", "finetune", "scratch"):
        wait_for_training_slot()
    if a.stage == "baseline":
        stage_baseline()
    elif a.stage == "cnn":
        stage_cnn(a.updates, a.lr, a.batch, a.channels, a.blocks)
    elif a.stage == "finetune":
        stage_finetune(a.kind, a.updates, a.lr, a.batch)
    elif a.stage == "scratch":
        stage_scratch(a.kind, a.updates, a.lr, a.batch, a.channels, a.blocks, a.original_trunk, a.seed, a.with_attack, a.with_quiet)
    else:
        stage_live(a.kind, a.seeds, a.games, a.candidate, a.name)


if __name__ == "__main__":
    main()
