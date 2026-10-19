"""Roll-out positions with a deep forced win (VCF of 3-4 fours) or a forbidden-block chain against black.

  deep          positions the exact 1-ply/2-ply teacher has no verdict for, where the side to move has a deep forced
                win (category 5, every first move of one) or the opponent has one (category 6, every legal move that
                leaves it no forced win by fours); both colors.
  forbid_black  black to move; white has a four whose only block cell is black-forbidden, so black must prevent it
                one move earlier (category 6, exact `tactics.analyze` labels).
Both are added to training moderately; the held-out slices (`deep_test`, `forbid_black_test`) use other seeds.
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

DEPTH = 4


def _deep_record(cells, color):
    """(labels, category) for a position with no exact verdict but a deep chain, else None."""
    labels, category = tactics.analyze(cells, color)
    if category:
        return None
    mine = tactics.deep_chains(cells, color, DEPTH)
    if mine:
        mask = np.zeros(CELLS, dtype=bool)
        mask[mine] = True
        return mask, tactics.CREATE_CHAIN
    if tactics.deep_chains(cells, -color, DEPTH):
        safe = tactics.deep_defence(cells, color, DEPTH)
        if safe.any():
            return safe, tactics.PREVENT_CHAIN
    return None


def _forbid_black_record(cells, color):
    if color != 1 or not rules.forbidden_mask(cells).any():
        return None
    chains = tactics.forcing_chains(cells, -1)
    if not any(follow == -1 for _, follow in chains.values()):
        return None
    labels, category = tactics.analyze(cells, 1)
    if category != tactics.PREVENT_CHAIN:
        return None
    return labels, category


def worker(args):
    mode, seed, quota, reserved, count, limit_seconds = args
    rng = np.random.default_rng(seed)
    board = BatchBoard(count)
    histories = np.full((count, CELLS), -1, dtype=np.int16)
    tactical_p = rng.uniform(0.5, 1.0, count)
    seen, items, scanned, started = set(reserved), [], 0, time.perf_counter()
    kept = {tactics.CREATE_CHAIN: 0, tactics.PREVENT_CHAIN: 0}
    half = max(1, quota // 2)
    while len(items) < quota and time.perf_counter() - started < limit_seconds:
        colors = board.to_move()
        threatened = has_threat_window(board)
        rows = np.flatnonzero(threatened & ~board.finished & (board.moves >= 8))
        if len(rows):
            obs = board.observe(colors)
        for row in rows:
            scanned += 1
            cells, color = board.board[row], int(colors[row])
            record = _forbid_black_record(cells, color) if mode == "forbid_black" else _deep_record(cells, color)
            if record is None:
                continue
            labels, category = record
            if mode == "deep" and kept[category] >= half:
                continue
            key = canonical(cells, color)
            if key in seen:
                continue
            seen.add(key)
            kept[category] = kept.get(category, 0) + 1
            items.append((obs[row].astype(np.int8), labels.copy(), rules.legal_mask(cells, color), category, key,
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
            tactical_p[done] = rng.uniform(0.5, 1.0, len(done))
    return items, scanned


def collect(mode, count, seed, reserved, workers, limit_seconds):
    quota = -(-count * 12 // (10 * workers))
    with mp.Pool(workers) as pool:
        parts = pool.map(worker, [(mode, seed + 7919 * w, quota, frozenset(reserved), 32, limit_seconds)
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
    parser.add_argument("mode", choices=("deep", "forbid_black"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--seed", type=int, default=991_000_001)
    args = parser.parse_args()
    prefix = args.prefix or args.mode
    reserved = set()
    for path in args.data.glob("*.npz"):
        with np.load(path) as saved:
            reserved |= set(saved["keys"].tolist())
    test, scanned = collect(args.mode, args.test, args.seed, reserved, args.workers, args.seconds)
    print("test", len(test), "scanned", scanned, {c: sum(r[3] == c for r in test) for c in (5, 6)}, flush=True)
    save(args.data / f"{prefix}_test.npz", test)
    train, scanned = collect(args.mode, args.train, args.seed + 1, reserved | {r[4] for r in test}, args.workers,
                             args.seconds)
    print("train", len(train), "scanned", scanned, {c: sum(r[3] == c for r in train) for c in (5, 6)}, flush=True)
    save(args.data / f"{prefix}_train.npz", train)


if __name__ == "__main__":
    main()
