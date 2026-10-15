"""Large exact-label defense datasets: synthetic roll-outs and on-policy visited positions.

Defense categories (labels come from the exact teachers in expanded/multistep tactics):
  2 unique block of an enemy four
  4 prevention of an enemy double-threat creation (includes open-three defense)
  6 safe defenses against an enemy forcing chain (exhaustive over every legal move)
"""
from pathlib import Path
import argparse
import json
import multiprocessing as mp

import numpy as np

from .env import BatchBoard, opponent_actions
from .midgame_data import canonical
from .multistep_tactics import multi_targets

DEFENSE = (2, 4, 6)
FIELDS = ("obs", "labels", "category", "keys", "histories")


def obs_key(obs):
    color = 1 if obs[3, 0, 0] else -1
    return canonical(((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8), color)


def _rollout_worker(args):
    seed, quota, reserved, cats = args
    rng = np.random.default_rng(seed)
    count = 128
    board = BatchBoard(count, 12)
    histories = np.full((count, 144), -1, dtype=np.int16)
    # Per-game mixture of tactical and random play gives diverse threat structures.
    tactical_p = rng.uniform(0.2, 0.95, count)
    rows_out = {c: [] for c in cats}
    seen = set(reserved)
    counts = {c: 0 for c in cats}
    for _ in range(100_000):
        colors = np.where(board.moves % 2 == 0, 1, -1).astype(np.int8)
        labels, category = multi_targets(board, colors)
        obs = board.observe(colors)
        for row in np.flatnonzero(np.isin(category, cats)):
            cat = int(category[row])
            if counts[cat] >= quota:
                continue
            key = canonical(board.board[row], int(colors[row]))
            if key in seen:
                continue
            seen.add(key)
            rows_out[cat].append((obs[row].astype(np.int8), labels[row].copy(), cat, key,
                                  histories[row].copy()))
            counts[cat] += 1
        if min(counts.values()) >= quota:
            break
        random_a = opponent_actions(board, colors, rng, "random")
        tactical_a = opponent_actions(board, colors, rng, "tactical")
        actions = np.where(rng.random(count) < tactical_p, tactical_a, random_a)
        histories[np.arange(count), board.moves] = actions
        board.place(actions, colors)
        done = np.flatnonzero(board.finished)
        board.reset(done)
        histories[done] = -1
        tactical_p[done] = rng.uniform(0.2, 0.95, len(done))
    return [item for c in cats for item in rows_out[c]]


def save(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, obs=np.stack([i[0] for i in items]), labels=np.stack([i[1] for i in items]),
        category=np.array([i[2] for i in items], dtype=np.int8), keys=np.array([i[3] for i in items]),
        histories=np.stack([i[4] for i in items]), round=np.zeros(len(items), dtype=np.int8), size=np.array(12))


def collect(path, per_category, seed, reserved, workers=5, cats=DEFENSE):
    if path.exists():
        with np.load(path) as saved:
            return set(saved["keys"].tolist())
    quota = -(-per_category // workers)
    with mp.Pool(workers) as pool:
        parts = pool.map(_rollout_worker, [(seed + 7919 * w, quota, frozenset(reserved), tuple(cats)) for w in range(workers)])
    items, seen = [], set(reserved)
    taken = {c: 0 for c in cats}
    for part in parts:
        for item in part:
            if item[3] in seen or taken[item[2]] >= per_category:
                continue
            seen.add(item[3])
            taken[item[2]] += 1
            items.append(item)
    save(path, items)
    print(f"{path.name}: {taken}", flush=True)
    return {i[3] for i in items}


def reserved_from_existing(root):
    """Canonical keys of the older supervised sets, so synthetic tests are unseen by the originals."""
    keys = set()
    for rel in ("runs/advanced_20261007/data/train.npz", "runs/advanced_20261007/data/validation.npz",
                "runs/advanced_20261007/data/test.npz", "runs/midgame_20261007/data/baseline_aux.npz",
                "runs/long_plan_f_20261009/defense_round3/gru_seed4810_dagger.npz"):
        path = root / rel
        if not path.exists():
            continue
        with np.load(path) as saved:
            for obs in saved["obs"]:
                keys.add(obs_key(obs))
    return keys


def on_policy_suite(replay_files, path):
    """Defense positions at the learner's own turns in real evaluation games (visited states)."""
    if path.exists():
        with np.load(path) as saved:
            return len(saved["keys"])
    seen, items, stats = set(), [], {}
    for file in replay_files:
        data = json.loads(file.read_text(encoding="utf-8"))
        replays = data["replays"]
        count = len(replays)
        board = BatchBoard(count, 12)
        histories = np.full((count, 144), -1, dtype=np.int16)
        lengths = np.array([len(r["moves"]) for r in replays])
        learner = np.array([r["a_color"] for r in replays])
        kind = replays[0]["a"]
        for t in range(int(lengths.max())):
            active = t < lengths
            colors = np.full(count, 1 if t % 2 == 0 else -1, dtype=np.int8)
            mine = active & (learner == colors)
            if mine.any():
                labels, category = multi_targets(board, colors)
                obs = board.observe(colors)
                for row in np.flatnonzero(mine & np.isin(category, DEFENSE) & ~board.finished):
                    key = canonical(board.board[row], int(colors[row]))
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append((obs[row].astype(np.int8), labels[row].copy(), int(category[row]), key,
                                  histories[row].copy()))
                    stats[(kind, int(category[row]))] = stats.get((kind, int(category[row])), 0) + 1
            moves = np.array([r["moves"][t] if t < len(r["moves"]) else 0 for r in replays])
            histories[np.flatnonzero(active), t] = moves[active]
            board.place(moves, colors, active)
    save(path, items)
    print("on-policy suite", len(items), {f"{k}:{c}": n for (k, c), n in sorted(stats.items())}, flush=True)
    return len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=800)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--attack", action="store_true", help="collect win/fork/forcing-chain categories instead")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=True)
    replay_files = sorted((root / "runs/balanced_curriculum_20261014").glob("*_original_*_*.json"))
    replay_files = [f for f in replay_files if f.name.split("_")[0] in ("fly", "gru", "rnn", "rewired")]
    if args.attack:
        old = reserved_from_existing(root)
        test_keys = collect(args.out / "attack_test.npz", args.test, 930_000_001, old, args.workers, (1, 3, 5))
        collect(args.out / "attack_train.npz", args.train, 940_000_001, old | test_keys, args.workers, (1, 3, 5))
        return
    on_policy_suite(replay_files, args.out / "onpolicy_test.npz")
    with np.load(args.out / "onpolicy_test.npz") as saved:
        onpolicy_keys = set(saved["keys"].tolist())
    old = reserved_from_existing(root)
    test_keys = collect(args.out / "synthetic_test.npz", args.test, 910_000_001, old | onpolicy_keys, args.workers)
    collect(args.out / "train.npz", args.train, 920_000_001, old | onpolicy_keys | test_keys, args.workers)


if __name__ == "__main__":
    main()
