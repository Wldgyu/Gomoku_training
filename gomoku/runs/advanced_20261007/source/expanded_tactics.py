"""Exact one-move double-winning-threat creation/prevention labels.

An edge (a,b) means a five-cell segment has three own stones and two empty
cells a,b, with no opponent stone. Playing at a makes b an immediate win.
Degree >=2 means two DISTINCT winning cells after that move.
Blocking a removes incident opponent edges; prevention requires every remaining
opponent vertex to have degree <=1. This includes open-three prevention.
"""
from pathlib import Path

import numpy as np

from .env import BatchBoard, opponent_actions
from .tactics import targets, position_key


def threat_graph(board: BatchBoard, colors):
    colors = np.broadcast_to(np.asarray(colors), (board.count,))
    rows = np.arange(board.count)
    slot = (colors == -1).astype(np.int64)
    own, other = board.line_counts[rows, slot], board.line_counts[rows, 1 - slot]
    rr, ll = np.nonzero((own == 3) & (other == 0))
    graph = np.zeros((board.count, board.size**2, board.size**2), dtype=bool)
    if len(rr):
        cells = board.lines[ll]
        empty_indexes = np.nonzero(board.board[rr[:, None], cells] == 0)[1].reshape(-1, 2)
        aa = cells[np.arange(len(rr)), empty_indexes[:, 0]]
        bb = cells[np.arange(len(rr)), empty_indexes[:, 1]]
        graph[rr, aa, bb] = True
        graph[rr, bb, aa] = True
    return graph


def expanded_targets(board, colors):
    colors = np.asarray(colors)
    base, category = targets(board, colors)
    # Basic categories always have priority, and an unanswerable enemy double
    # immediate win is not mislabelled as preventable.
    enemy_wins, enemy_category = targets(board, -colors)
    no_immediate_enemy = ~((enemy_category == 1) & enemy_wins.any(1))
    available = (category == 0) & no_immediate_enemy & ~board.finished
    attack = threat_graph(board, colors).sum(-1, dtype=np.int16) >= 2
    attack &= board.legal()
    choose_attack = available & attack.any(1)
    base[choose_attack] = attack[choose_attack]
    category[choose_attack] = 3
    available &= ~choose_attack
    enemy = threat_graph(board, -colors)
    degrees = enemy.sum(-1, dtype=np.int16)
    remaining = degrees[:, :, None] - enemy
    diag = np.arange(board.size**2)
    remaining[:, diag, diag] = 0
    prevent = (remaining.max(axis=1) < 2) & board.legal()
    choose_prevent = available & (degrees >= 2).any(1) & prevent.any(1)
    base[choose_prevent] = prevent[choose_prevent]
    category[choose_prevent] = 4
    return base, category


def generate(path: Path, per_category, seed, probability, excluded=None):
    if path.exists():
        with np.load(path) as d:
            return set(d["keys"].tolist())
    rng = np.random.default_rng(seed)
    board = BatchBoard(128, 12)
    histories = np.full((128, 144), -1, dtype=np.int16)
    obs, labels, categories, keys, saved_histories = [], [], [], [], []
    seen = set() if excluded is None else set(excluded)
    counts = {c: 0 for c in (1, 2, 3, 4)}
    for iteration in range(100_000):
        colors = np.where(board.moves % 2 == 0, 1, -1).astype(np.int8)
        target, category = expanded_targets(board, colors)
        observed = board.observe(colors)
        for row in np.flatnonzero(category):
            cat = int(category[row])
            if counts[cat] >= per_category:
                continue
            key = position_key(board.board[row], int(colors[row]))
            if key in seen:
                continue
            seen.add(key)
            obs.append(observed[row].astype(np.int8))
            labels.append(target[row])
            categories.append(cat)
            keys.append(key)
            saved_histories.append(histories[row].copy())
            counts[cat] += 1
        if min(counts.values()) >= per_category:
            break
        if iteration % 500 == 0:
            print(f"collect {path.name} step={iteration} counts={counts}", flush=True)
        random = opponent_actions(board, colors, rng, "random")
        tactical = opponent_actions(board, colors, rng, "tactical")
        actions = np.where(rng.random(board.count) < probability, tactical, random)
        histories[np.arange(board.count), board.moves] = actions
        board.place(actions, colors)
        done = np.flatnonzero(board.finished)
        board.reset(done)
        histories[done] = -1
    else:
        raise RuntimeError(f"Insufficient expanded tactical positions: {counts}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=np.array(obs), labels=np.array(labels), category=np.array(categories),
                        keys=np.array(keys), histories=np.array(saved_histories), size=12, seed=seed,
                        tactical_probability=probability)
    print(f"dataset {path.name}: {counts}, steps={iteration}", flush=True)
    return set(keys)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    a = p.parse_args()
    old_keys, old_test_validation = set(), set()
    for name in ("train", "validation", "test"):
        with np.load(a.previous / f"{name}.npz") as d:
            ks = set(d["keys"].tolist())
            old_keys |= ks
            if name != "train":
                old_test_validation |= ks
    test = generate(a.output / "test.npz", 512, 112_000_012, 0.8, old_keys)
    validation = generate(a.output / "validation.npz", 256, 111_000_012, 0.5, old_keys | test)
    generate(a.output / "train.npz", 5000, 110_000_012, 0.5, old_test_validation | test | validation)


if __name__ == "__main__":
    main()
