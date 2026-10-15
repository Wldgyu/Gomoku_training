"""Quiet opening/middle-game positions labelled by Rapfi (15x15 Renju rule, NNUE weights, 200 ms search).

Positions come from mixed random/rule roll-outs (source `rollout`) or sampled self-play of a trained policy
(source `selfplay`) and are kept when the exact teacher has no label and nobody has an immediate win.
The label is the engine's suggestion (a search recommendation, not a proof); category 7.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from ..models import load_policy
from . import rules, tactics
from .data import canonical, has_threat_window, obs_key, save
from .env import BatchBoard, force_black_center, opponent_actions
from .rules import CELLS, SIZE

ENGINE = Path(__file__).resolve().parents[1] / "external/rapfi_250615/engine"
QUIET = 7
RENJU_RULE = 4


def _quiet_row(board, row, color):
    cells = board.board[row]
    if rules.winning_cells(cells, color).any() or rules.winning_cells(cells, -color).any():
        return False
    return tactics.analyze(cells, color)[1] == 0


def _record(board, histories, row, color, colors_obs):
    return (colors_obs[row].astype(np.int8), np.zeros(CELLS, dtype=bool), rules.legal_mask(board.board[row], color),
            QUIET, canonical(board.board[row], color), histories[row].copy())


def rollout_quiet(args):
    seed, quota, reserved, count, plies, keep, limit_seconds = args
    rng = np.random.default_rng(seed)
    board = BatchBoard(count)
    histories = np.full((count, CELLS), -1, dtype=np.int16)
    tactical_p = rng.uniform(0.2, 0.95, count)
    seen, items, started = set(reserved), [], time.perf_counter()
    while len(items) < quota and time.perf_counter() - started < limit_seconds:
        colors = board.to_move()
        eligible = (board.moves >= plies[0]) & (board.moves <= plies[1]) & ~board.finished & (rng.random(count) < keep)
        if eligible.any():
            obs = board.observe(colors)
            threatened = has_threat_window(board)
            for row in np.flatnonzero(eligible):
                color = int(colors[row])
                if threatened[row] and not _quiet_row(board, row, color):
                    continue
                key = canonical(board.board[row], color)
                if key in seen:
                    continue
                seen.add(key)
                items.append(_record(board, histories, row, color, obs))
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
    return items


@torch.no_grad()
def selfplay_quiet(model_path, games, seed, quota, reserved, plies=(4, 60), keep=0.2, temperature=1.0,
                   tactical_mix=0.15, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, _ = load_policy(Path(model_path), device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    board = BatchBoard(games)
    histories = np.full((games, CELLS), -1, dtype=np.int16)
    states = {1: model.initial(games, device), -1: model.initial(games, device)}
    zeros = torch.zeros(games, dtype=torch.bool, device=device)
    matrix = model.matrix()
    seen, items = set(reserved), []
    while len(items) < quota:
        colors = board.to_move()
        obs_np = board.observe(colors)
        legal = torch.from_numpy(board.legal(colors)).to(device)
        mine = torch.from_numpy(colors == 1).to(device)[:, None]
        state = torch.where(mine, states[1], states[-1])
        logits, _, new_state = model.step(torch.from_numpy(obs_np).to(device), state, zeros, matrix)
        states[1] = torch.where(mine, new_state, states[1])
        states[-1] = torch.where(mine, states[-1], new_state)
        probs = torch.softmax(logits.masked_fill(~legal, -1e9) / temperature, 1)
        actions = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()
        tactical = opponent_actions(board, colors, rng, "tactical")
        actions = force_black_center(board, colors, np.where(rng.random(games) < tactical_mix, tactical, actions))
        eligible = (board.moves >= plies[0]) & (board.moves <= plies[1]) & ~board.finished & (rng.random(games) < keep)
        threatened = has_threat_window(board)
        for row in np.flatnonzero(eligible):
            color = int(colors[row])
            if threatened[row] and not _quiet_row(board, row, color):
                continue
            key = canonical(board.board[row], color)
            if key in seen:
                continue
            seen.add(key)
            items.append(_record(board, histories, row, color, obs_np))
        rows = np.flatnonzero(~board.finished)
        histories[rows, board.moves[rows]] = actions[rows]
        board.place(actions, colors)
        done = np.flatnonzero(board.finished)
        if len(done):
            board.reset(done)
            histories[done] = -1
            for color in (1, -1):
                states[color][torch.from_numpy(done).to(device)] = 0
    return items[:quota]


_engine = None


def _label(task):
    global _engine
    from ..external_engine import PiskvorkEngine
    if _engine is None:
        _engine = PiskvorkEngine([ENGINE / "pbrain-rapfi-windows-sse.exe"], size=SIZE, turn_ms=task[2], cwd=ENGINE,
                                 rule=RENJU_RULE)
    obs, history = task[0], task[1]
    color = 1 if obs[3, 0, 0] else -1
    cells = ((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8)
    response = _engine.recommend(cells, color, history=history)
    return response["action"], response["evaluation"], response["seconds"]


def build(path, items, turn_ms, workers, source):
    if path.exists():
        with np.load(path) as saved:
            return set(saved["keys"].tolist())
    print(f"{path.name}: {len(items)} positions to label", flush=True)
    with mp.Pool(workers) as pool:
        answers = pool.map(_label, [(i[0], i[5], turn_ms) for i in items], chunksize=16)
    labelled, rejected = [], 0
    for item, (action, evaluation, seconds) in zip(items, answers):
        if not item[2][action]:  # the engine suggested a cell our rules treat as illegal (e.g. forbidden)
            rejected += 1
            continue
        labels = np.zeros(CELLS, dtype=bool)
        labels[action] = True
        labelled.append((item[0], labels, item[2], QUIET, item[4], item[5]))
    save(path, labelled)
    path.with_suffix(".json").write_text(json.dumps(dict(
        teacher="Rapfi 250615 renju NNUE + search", rule="renju", size=SIZE, turn_ms=turn_ms, source=source,
        records=len(labelled), rejected_illegal=rejected, evaluations=[a[1] for a in answers],
        scope="search suggestions on quiet states, not forced-win certificates"), indent=1))
    print(f"{path.name}: labelled {len(labelled)}, rejected {rejected}", flush=True)
    return {r[4] for r in labelled}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None, help="policy for self-play positions (default: roll-outs)")
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--turn-ms", type=int, default=200)
    parser.add_argument("--prefix", default="quiet")
    args = parser.parse_args()
    reserved = set()
    for name in ("train", "test", "onpolicy_test", "onpolicy_train"):
        if (args.data / f"{name}.npz").exists():
            with np.load(args.data / f"{name}.npz") as saved:
                reserved |= set(saved["keys"].tolist())

    def generate(count, seed, excluded):
        if args.model:
            return selfplay_quiet(args.model, 128, seed, count, excluded)
        quota = -(-count * 12 // (10 * args.workers))
        with mp.Pool(args.workers) as pool:
            parts = pool.map(rollout_quiet, [(seed + 7919 * w, quota, frozenset(excluded), 32, (4, 60), 0.15, 1800)
                                             for w in range(args.workers)])
        out, seen = [], set(excluded)
        for part in parts:
            for item in part:
                if item[4] not in seen:
                    seen.add(item[4])
                    out.append(item)
        return out[:count]

    source = f"selfplay:{args.model}" if args.model else "rollout"
    test_items = generate(args.test, 950_000_001, reserved)
    test_keys = build(args.data / f"{args.prefix}_test.npz", test_items, args.turn_ms, args.workers, source)
    train_items = generate(args.train, 960_000_001, reserved | test_keys)
    build(args.data / f"{args.prefix}_train.npz", train_items, args.turn_ms, args.workers, source)


if __name__ == "__main__":
    main()
