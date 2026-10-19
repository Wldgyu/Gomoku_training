"""Positions where white wins by a four that black may not block (its only blocking cell is forbidden).

Found by scanning roll-outs for white-to-move positions with such a chain (`tactics.forcing_chains` entry whose
follow-up is -1) that the exact teacher files under category 5 (no immediate fork / block duty). The label set is
every exact chain move, so ordinary forcing chains stay correct answers too. Kept in moderation on purpose.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import rules, tactics
from .data import canonical, has_threat_window, save
from .env import BatchBoard, force_black_center, opponent_actions
from .rules import CELLS

CHAIN = 5


def worker(args):
    seed, quota, reserved, count, limit_seconds = args
    rng = np.random.default_rng(seed)
    board = BatchBoard(count)
    histories = np.full((count, CELLS), -1, dtype=np.int16)
    tactical_p = rng.uniform(0.3, 0.95, count)
    seen, items, scanned, started = set(reserved), [], 0, time.perf_counter()
    while len(items) < quota and time.perf_counter() - started < limit_seconds:
        colors = board.to_move()
        threatened = has_threat_window(board)
        rows = np.flatnonzero((colors == -1) & threatened & ~board.finished & (board.moves >= 10))
        if len(rows):
            obs = board.observe(colors)
        for row in rows:
            scanned += 1
            cells = board.board[row]
            if not rules.forbidden_mask(cells).any():
                continue
            chains = tactics.forcing_chains(cells, -1)
            if not any(follow == -1 for _, follow in chains.values()):
                continue
            labels, category = tactics.analyze(cells, -1)
            if category != CHAIN:
                continue
            key = canonical(cells, -1)
            if key in seen:
                continue
            seen.add(key)
            items.append((obs[row].astype(np.int8), labels.copy(), rules.legal_mask(cells, -1), CHAIN, key,
                          histories[row].copy()))
        random_a = opponent_actions(board, colors, rng, "random")
        tactical_a = opponent_actions(board, colors, rng, "tactical")
        actions = force_black_center(board, colors, np.where(rng.random(count) < tactical_p, tactical_a, random_a))
        live = np.flatnonzero(~board.finished)
        histories[live, board.moves[live]] = actions[live]
        board.place(actions, colors)
        done = np.flatnonzero(board.finished)
        if len(done):
            board.reset(done)
            histories[done] = -1
            tactical_p[done] = rng.uniform(0.3, 0.95, len(done))
    return items, scanned


def collect(count, seed, reserved, workers, limit_seconds):
    quota = -(-count * 12 // (10 * workers))
    with mp.Pool(workers) as pool:
        parts = pool.map(worker, [(seed + 7919 * w, quota, frozenset(reserved), 32, limit_seconds)
                                  for w in range(workers)])
    out, seen, scanned = [], set(reserved), 0
    for items, n in parts:
        scanned += n
        for item in items:
            if item[4] not in seen:
                seen.add(item[4])
                out.append(item)
    return out[:count], scanned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=990_000_001)
    parser.add_argument("--prefix", default="forbid")
    args = parser.parse_args()
    reserved = set()
    for path in args.data.glob("*.npz"):
        with np.load(path) as saved:
            reserved |= set(saved["keys"].tolist())
    test, scanned = collect(args.test, args.seed, reserved, args.workers, args.seconds)
    print("test", len(test), "scanned", scanned, flush=True)
    save(args.data / f"{args.prefix}_test.npz", test)
    train, scanned = collect(args.train, args.seed + 1, reserved | {r[4] for r in test}, args.workers, args.seconds)
    print("train", len(train), "scanned", scanned, flush=True)
    save(args.data / f"{args.prefix}_train.npz", train)


if __name__ == "__main__":
    main()
