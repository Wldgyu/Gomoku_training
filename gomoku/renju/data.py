"""Exact-label tactical datasets for 15x15 Renju from mixed random/rule roll-outs.

Every record: obs (4,15,15) int8, labels (225) bool (a set of equally good moves), legal (225) bool (black's
forbidden cells removed), category 1..6, a D4-canonical key (positions are never shared between splits),
and the full move history that led to the position (needed for recurrent state burn-in).
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import rules, tactics
from .env import BatchBoard, force_black_center, opponent_actions
from .rules import CELLS, SIZE, WINDOWS

FIELDS = ("obs", "labels", "legal", "category", "keys", "histories")


def canonical(cells, color):
    grid = np.asarray(cells, dtype=np.int8).reshape(SIZE, SIZE)
    best = None
    for s in range(8):
        g = grid[:, ::-1] if s >= 4 else grid
        data = np.rot90(g, s % 4).tobytes()
        best = data if best is None or data < best else best
    return hashlib.sha1(best + bytes([color + 1])).hexdigest()


def obs_key(obs):
    color = 1 if obs[3, 0, 0] else -1
    return canonical(((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8), color)


def position_obs(cells, color, last):
    """int8 (4,15,15) observation for the side `color` to move (mine, other, last move, black flag)."""
    return np.stack((cells == color, cells == -color, np.arange(CELLS) == last,
                     np.full(CELLS, color == 1))).astype(np.int8).reshape(4, SIZE, SIZE)


def has_threat_window(board):
    """Rows where some side owns >= 3 stones in a five-window free of the other side."""
    values = board.board[:, WINDOWS]
    out = np.zeros(board.count, dtype=bool)
    for color in (1, -1):
        own = (values == color).sum(2)
        other = (values == -color).sum(2)
        out |= ((own >= 3) & (other == 0)).any(1)
    return out


def rollout(args):
    seed, quota, reserved, cats, count, limit_seconds = args
    rng = np.random.default_rng(seed)
    board = BatchBoard(count)
    histories = np.full((count, CELLS), -1, dtype=np.int16)
    tactical_p = rng.uniform(0.2, 0.95, count)
    items = {c: [] for c in cats}
    seen = set(reserved)
    started = time.perf_counter()
    stats = dict(positions=0, analysed=0)
    while time.perf_counter() - started < limit_seconds and min(len(v) for v in items.values()) < quota:
        colors = board.to_move()
        threatened = has_threat_window(board) & ~board.finished & (board.moves >= 5)
        stats["positions"] += int((~board.finished).sum())
        for row in np.flatnonzero(threatened):
            stats["analysed"] += 1
            color = int(colors[row])
            labels, category = tactics.analyze(board.board[row], color)
            if category not in items or len(items[category]) >= quota or not labels.any():
                continue
            key = canonical(board.board[row], color)
            if key in seen:
                continue
            seen.add(key)
            cells = board.board[row]
            obs = np.stack((cells == color, cells == -color, np.arange(CELLS) == board.last[row],
                            np.full(CELLS, color == 1))).astype(np.int8).reshape(4, SIZE, SIZE)
            legal = rules.legal_mask(cells, color)
            items[category].append((obs, labels.copy(), legal, category, key, histories[row].copy()))
        random_a = opponent_actions(board, colors, rng, "random")
        tactical_a = opponent_actions(board, colors, rng, "tactical")
        actions = force_black_center(board, colors, np.where(rng.random(count) < tactical_p, tactical_a, random_a))
        rows = np.flatnonzero(~board.finished)
        histories[rows, board.moves[rows]] = actions[rows]
        board.place(actions, colors)
        done = np.flatnonzero(board.finished)
        if len(done):
            board.reset(done)
            histories[done] = -1
            tactical_p[done] = rng.uniform(0.2, 0.95, len(done))
    return items, stats


def save(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, obs=np.stack([r[0] for r in records]), labels=np.stack([r[1] for r in records]),
        legal=np.stack([r[2] for r in records]), category=np.array([r[3] for r in records], dtype=np.int8),
        keys=np.array([r[4] for r in records]), histories=np.stack([r[5] for r in records]), size=np.array(SIZE))


def collect(path, per_category, seed, reserved=frozenset(), workers=10, cats=(1, 2, 3, 4, 5, 6),
            count=32, limit_seconds=3600):
    if path.exists():
        with np.load(path) as saved:
            return set(saved["keys"].tolist())
    quota = -(-per_category * 12 // (10 * workers))  # a little surplus; duplicates are dropped below
    with mp.Pool(workers) as pool:
        parts = pool.map(rollout, [(seed + 7919 * w, quota, frozenset(reserved), tuple(cats), count, limit_seconds)
                                   for w in range(workers)])
    taken = {c: 0 for c in cats}
    seen, records = set(reserved), []
    total = dict(positions=sum(p[1]["positions"] for p in parts), analysed=sum(p[1]["analysed"] for p in parts))
    for items, _ in parts:
        for c in cats:
            for record in items[c]:
                if record[4] in seen or taken[c] >= per_category:
                    continue
                seen.add(record[4])
                taken[c] += 1
                records.append(record)
    save(path, records)
    print(f"{path.name}: {taken} from {total}", flush=True)
    return {r[4] for r in records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=600)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--minutes", type=int, default=120)
    args = parser.parse_args()
    seconds = args.minutes * 60
    test_keys = collect(args.out / "test.npz", args.test, 710_000_001, workers=args.workers, limit_seconds=seconds)
    collect(args.out / "train.npz", args.train, 720_000_001, test_keys, workers=args.workers, limit_seconds=seconds)


if __name__ == "__main__":
    main()
