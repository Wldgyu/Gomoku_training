"""Exact-label positions the trained policies actually reach (games against the rule opponent).

The learner's own turns are scanned with the exact teacher. Positions it handled badly are what the next
training round needs, and a held-out slice (`onpolicy_test`) measures defense in realistic play.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..models import load_policy
from . import rules, tactics
from .data import canonical, has_threat_window, obs_key, save
from .env import LearnerEnv, force_black_center
from .rules import CELLS


@torch.no_grad()
def collect_positions(model, games, seed, steps, reserved=frozenset(), opponent="tactical", device="cpu",
                      categories=(1, 2, 3, 4, 5, 6), wrong_only=False):
    env = LearnerEnv(games, seed, opponent)
    dev = torch.device(device)
    model = model.to(dev).eval()
    histories = np.full((games, CELLS), -1, dtype=np.int16)
    for row in np.flatnonzero(env.colors == -1):
        histories[row, 0] = env.games.last[row]
    state = model.initial(games, dev)
    resets = torch.ones(games, dtype=torch.bool, device=dev)
    matrix = model.matrix()
    seen, records, stats = set(reserved), [], {}
    for _ in range(steps):
        obs, legal = env.observe()
        logits, _, state = model.step(torch.from_numpy(obs).to(dev), state, resets, matrix)
        action = logits.masked_fill(~torch.from_numpy(legal).to(dev), -1e9).argmax(1).cpu().numpy()
        action = force_black_center(env.games, env.colors, action)
        for row in np.flatnonzero(has_threat_window(env.games) & (env.games.moves >= 5)):
            color = int(env.colors[row])
            labels, category = tactics.analyze(env.games.board[row], color)
            if category not in categories or not labels.any():
                continue
            if wrong_only and labels[action[row]]:
                continue
            key = canonical(env.games.board[row], color)
            if key in seen:
                continue
            seen.add(key)
            stats[category] = stats.get(category, 0) + 1
            records.append((obs[row].astype(np.int8), labels.copy(), legal[row].copy(), category, key,
                            histories[row].copy()))
        before = env.games.moves.copy()
        histories[np.arange(games), before] = action
        _, done, info = env.step(action)
        for row in np.flatnonzero(info["reply_active"]):
            histories[row, before[row] + 1] = info["replies"][row]
        for row in np.flatnonzero(done):
            histories[row] = -1
            if env.colors[row] == -1:
                histories[row, 0] = env.games.last[row]
        resets = torch.from_numpy(done).to(dev)
    return records, stats


@torch.no_grad()
def collect_selfplay(model_a, model_b, games, seed, reserved=frozenset(), temperature=0.8, device="cpu",
                     keep_right=0.25):
    """Positions from model-vs-model games (random 4-ply openings, sampled moves after them).

    Both colors' positions are labelled exactly. The mover's greedy choice is checked, so positions where
    the policy errs ("hard" ones) are always kept and correctly handled ones are kept with `keep_right`.
    Returns (records, stats[category] = [kept, mover_correct, mover_wrong]).
    """
    from .env import BatchBoard
    from .league import OPENING_PLIES, make_openings
    dev = torch.device(device)
    models = (model_a.to(dev).eval(), model_b.to(dev).eval())
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    openings = np.array(make_openings(games, seed))
    board = BatchBoard(games)
    histories = np.full((games, CELLS), -1, dtype=np.int16)
    a_black = np.arange(games) % 2 == 0
    states = [m.initial(games, dev) for m in models]
    matrices = [m.matrix() for m in models]
    zeros = torch.zeros(games, dtype=torch.bool, device=dev)
    seen, records, stats = set(reserved), [], {}
    for ply in range(CELLS):
        if board.finished.all():
            break
        colors = board.to_move()
        a_turn = (colors == 1) == a_black
        obs = board.observe(colors)
        legal = board.legal(colors)
        obs_t, legal_t = torch.from_numpy(obs).to(dev), torch.from_numpy(legal).to(dev)
        greedy, sampled = [], []
        for k, model in enumerate(models):
            logits, _, new = model.step(obs_t, states[k], zeros, matrices[k])
            mover = torch.from_numpy(a_turn if k == 0 else ~a_turn).to(dev)[:, None]
            states[k] = torch.where(mover, new, states[k])
            masked = logits.masked_fill(~legal_t, -1e9)
            greedy.append(masked.argmax(1).cpu().numpy())
            sampled.append(torch.multinomial(torch.softmax(masked / temperature, 1), 1).squeeze(1).cpu().numpy())
        greedy_action = np.where(a_turn, greedy[0], greedy[1])
        action = np.where(a_turn, sampled[0], sampled[1])
        if ply < OPENING_PLIES:
            action = openings[:, ply]
        else:
            for row in np.flatnonzero(has_threat_window(board) & ~board.finished):
                color = int(colors[row])
                labels, category = tactics.analyze(board.board[row], color)
                if not category or not labels.any():
                    continue
                right = bool(labels[greedy_action[row]])
                entry = stats.setdefault(category, [0, 0, 0])
                entry[1 if right else 2] += 1
                if right and rng.random() >= keep_right:
                    continue
                key = canonical(board.board[row], color)
                if key in seen:
                    continue
                seen.add(key)
                entry[0] += 1
                records.append((obs[row].astype(np.int8), labels.copy(), legal[row].copy(), category, key,
                                histories[row].copy()))
        rows = np.flatnonzero(~board.finished)
        histories[rows, board.moves[rows]] = action[rows]
        board.place(action, colors, ~board.finished)
    return records, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfplay", action="store_true", help="model-vs-model positions (needs exactly 2 models)")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True, help="directory with train.npz/test.npz (keys are excluded)")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--wrong-only", action="store_true", help="keep only positions the policy got wrong")
    parser.add_argument("--rounds", type=int, default=4, help="self-play rounds for the training slice")
    parser.add_argument("--name", default="onpolicy")
    parser.add_argument("--seed", type=int, default=770_000_001)
    args = parser.parse_args()
    torch.set_num_threads(3)
    reserved = set()
    for name in ("train", "test", "quiet_train", "quiet_test"):
        path = args.data / f"{name}.npz"
        if path.exists():
            with np.load(path) as saved:
                reserved |= set(saved["keys"].tolist())
    test_records, train_records = [], []
    if args.selfplay:
        models = [load_policy(p, torch.device("cpu"))[0] for p in args.models]
        for part, bucket, seed in (("test", test_records, args.seed), ("train", train_records, args.seed + 1000)):
            for round_ in range(args.rounds if part == "train" else 1):
                seen = reserved | {r[4] for r in test_records + train_records}
                records, stats = collect_selfplay(models[0], models[1], args.games, seed + round_, seen)
                bucket.extend(records)
                print(part, round_, len(records), stats, flush=True)
        save(args.data / f"{args.name}_test.npz", test_records)
        save(args.data / f"{args.name}_train.npz", train_records)
        return
    for i, path in enumerate(args.models):
        model, _ = load_policy(path, torch.device("cpu"))
        for part, bucket, seed in (("test", test_records, args.seed + i), ("train", train_records, args.seed + 1000 + i)):
            seen = reserved | {r[4] for r in test_records + train_records}
            records, stats = collect_positions(model, args.games, seed, args.steps, seen, wrong_only=args.wrong_only and part == "train")
            bucket.extend(records)
            print(path.stem, part, len(records), stats, flush=True)
    save(args.data / f"{args.name}_test.npz", test_records)
    save(args.data / f"{args.name}_train.npz", train_records)


if __name__ == "__main__":
    main()
