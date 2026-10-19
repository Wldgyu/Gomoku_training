"""Rapfi DAgger: positions the policy reaches against a strong engine, labelled by the exact teacher or Rapfi.

The policy plays whole games against Rapfi (renju rule). Every position where the policy is to move is
labelled: with the exact teacher when it has a verdict (categories 1-6), otherwise with Rapfi's suggestion
(category 7; a search recommendation, not a proof). `beta` is the share of the policy's own moves that are
replaced by the label, so the games also reach positions the policy would already have lost.

Modes:
  collect   policy-vs-Rapfi games -> <prefix>_train.npz / <prefix>_test.npz (test = other seeds, kept apart)
  eval      greedy policy vs Rapfi, results only (no labelling)
  human     label the policy's moves in saved human-vs-policy games (no held-out split)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from . import rules, tactics
from .data import canonical, has_threat_window, save
from .env import BatchBoard, opponent_actions
from .quiet import ENGINE, QUIET, RENJU_RULE
from .rules import CELLS, CENTER, SIZE

OPENING_RADIUS = 2
RELABEL_LAST = 10  # lost games: engine-labelled positions among the policy's last turns that get the long relabel


def _engine(turn_ms):
    from ..external_engine import PiskvorkEngine
    return PiskvorkEngine([ENGINE / "pbrain-rapfi-windows-sse.exe"], size=SIZE, turn_ms=turn_ms, cwd=ENGINE,
                          rule=RENJU_RULE)


def _set_time(engine, turn_ms):
    engine.send(f"INFO timeout_turn {turn_ms}")
    engine.turn_ms = turn_ms


def label_position(cells, color, moves, engine, legal=None, deep=True):
    """(labels, category) from the exact teacher (plus deep VCF chains when `deep`), else from Rapfi; None if neither."""
    legal = rules.legal_mask(cells, color) if legal is None else legal
    labels, category = None, 0
    board = BatchBoard(1)
    board.board[0] = cells
    if has_threat_window(board)[0]:
        labels, category = tactics.analyze_deep(cells, color) if deep else tactics.analyze(cells, color)
        category = int(category)
        if not category or not labels.any():
            labels, category = None, 0
    if labels is not None:
        return labels, category
    action = engine.recommend(cells, color, history=moves)["action"]
    if not legal[action]:
        return None
    labels = np.zeros(CELLS, dtype=bool)
    labels[action] = True
    return labels, QUIET


def _opening_cells(board, rng):
    row, column = divmod(CENTER, SIZE)
    cells = [r * SIZE + c for r in range(row - OPENING_RADIUS, row + OPENING_RADIUS + 1)
             for c in range(column - OPENING_RADIUS, column + OPENING_RADIUS + 1) if board[r * SIZE + c] == 0]
    return int(rng.choice(cells))


def hunter_move(cells, color, rng):
    """Forced-win move for the opponent (exact teacher categories 1/3/5 or a deep VCF), else None."""
    labels, category = tactics.analyze(cells, color)
    if category in (tactics.WIN, tactics.CREATE_FORK, tactics.CREATE_CHAIN) and labels.any():
        return int(rng.choice(np.flatnonzero(labels)))
    deep = tactics.deep_chains(cells, color) if not category else []
    return int(rng.choice(deep)) if deep else None


def play_game(model, model_color, opponent, labeller, rng, beta, temperature, opening_plies, reserved, label=True,
              blunder=0.0, hunt=0.0, relabel_ms=0, label_ms=150):
    """One game. Returns (records, result) with result = +1 policy win, -1 loss, 0 draw/void.

    `hunt`: chance that the opponent plays a forced win (exact teacher or deep VCF) whenever it has one.
    `relabel_ms`: if the policy loses, its last few engine-labelled positions are relabelled with a longer search.
    """
    board = BatchBoard(1)
    histories = np.full(CELLS, -1, dtype=np.int16)
    moves, records, seen = [], [], set(reserved)
    contexts = []
    state = model.initial(1, torch.device("cpu"))
    matrix = model.matrix()
    zero = torch.zeros(1, dtype=torch.bool)
    result = 0
    for ply in range(CELLS):
        color = 1 if ply % 2 == 0 else -1
        cells = board.board[0]
        legal = rules.legal_mask(cells, color)
        if color == model_color:
            obs = board.observe(color)
            with torch.no_grad():
                logits, _, state = model.step(torch.from_numpy(obs), state, zero, matrix)
            masked = logits.masked_fill(~torch.from_numpy(legal)[None], -1e9)[0]
            if temperature > 0:
                action = int(torch.multinomial(torch.softmax(masked / temperature, 0), 1))
            else:
                action = int(masked.argmax())
            if ply == 0:
                action = CENTER
            elif ply >= 4 and label:
                key = canonical(cells, color)
                found = None if key in seen else label_position(cells, color, moves, labeller, legal)
                if found is not None:
                    seen.add(key)
                    labels, category = found
                    records.append((obs[0].astype(np.int8), labels, legal.copy(), category, key, histories.copy()))
                    contexts.append((cells.copy(), color, list(moves), legal.copy()))
                    if ply >= opening_plies and rng.random() < beta:
                        action = int(rng.choice(np.flatnonzero(labels)))
        elif ply == 0:
            action = CENTER
        elif ply < opening_plies:
            action = _opening_cells(cells, rng)
        elif rng.random() < blunder:  # weaker opponent turn (rule AI) so games last longer and vary
            action = int(opponent_actions(board, np.array([color], dtype=np.int8), rng, "tactical")[0])
        else:
            action = hunter_move(cells, color, rng) if hunt and rng.random() < hunt else None
            if action is None:
                action = opponent.recommend(cells, color, history=moves)["action"]
            if not legal[action]:  # engine and our forbidden-move rules disagree: void the game
                return records, 0
        if ply < opening_plies and color == model_color and ply > 0:
            action = _opening_cells(cells, rng)
        histories[ply] = action
        moves.append(action)
        board.place(np.array([action]), np.array([color], dtype=np.int8))
        if board.finished[0]:
            result = int(board.winner[0]) * model_color
            break
    if result == -1 and relabel_ms and label and labeller is not None:
        _set_time(labeller, relabel_ms)
        try:
            for index in range(max(0, len(records) - RELABEL_LAST), len(records)):
                if records[index][3] != QUIET:
                    continue
                cells, color, history, legal = contexts[index]
                action = labeller.recommend(cells, color, history=history)["action"]
                if legal[action]:
                    labels = np.zeros(CELLS, dtype=bool)
                    labels[action] = True
                    records[index] = records[index][:1] + (labels,) + records[index][2:]
        finally:
            _set_time(labeller, label_ms)
    return records, result


def _worker(args):
    (model_path, seed, games, betas, temperature, opp_ms, label_ms, reserved, label, blunders, hunts, relabel_ms) = args
    torch.set_num_threads(1)
    from ..models import load_policy
    model = load_policy(Path(model_path), torch.device("cpu"))[0].eval()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opponent, labeller = _engine(opp_ms[0]), (_engine(label_ms) if label else None)
    records, results = [], []
    try:
        for g in range(games):
            _set_time(opponent, int(rng.choice(opp_ms)))
            beta = float(rng.choice(betas))
            opening = int(rng.integers(3, 6))
            blunder = float(rng.choice(blunders))
            hunt = float(rng.choice(hunts))
            out, result = play_game(model, 1 if g % 2 == 0 else -1, opponent, labeller, rng, beta, temperature,
                                    opening, reserved, label, blunder, hunt, relabel_ms, label_ms)
            records.extend(out)
            results.append((Path(model_path).stem, 1 if g % 2 == 0 else -1, result, beta))
    finally:
        opponent.close()
        if labeller:
            labeller.close()
    return records, results


def run(models, games, seed, workers, betas, temperature, opp_ms, label_ms, reserved, label=True, blunders=(0.0,),
        hunts=(0.0,), relabel_ms=0):
    jobs = [(str(models[w % len(models)]), seed + 7919 * w, -(-games // workers), betas, temperature,
             opp_ms, label_ms, frozenset(reserved), label, blunders, hunts, relabel_ms) for w in range(workers)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_worker, jobs)
    records, results, seen = [], [], set(reserved)
    for part_records, part_results in parts:
        results.extend(part_results)
        for r in part_records:
            if r[4] not in seen:
                seen.add(r[4])
                records.append(r)
    return records, results


def summarize(results):
    out = {}
    for name in sorted({r[0] for r in results}):
        rows = [r for r in results if r[0] == name]
        out[name] = {"games": len(rows), "wins": sum(r[2] == 1 for r in rows), "draws": sum(r[2] == 0 for r in rows),
                     "losses": sum(r[2] == -1 for r in rows),
                     "as_black_wins": sum(r[2] == 1 for r in rows if r[1] == 1),
                     "as_white_wins": sum(r[2] == 1 for r in rows if r[1] == -1)}
    return out


def keys_of(directory, names):
    keys = set()
    for name in names:
        path = directory / f"{name}.npz"
        if path.exists():
            with np.load(path) as saved:
                keys |= set(saved["keys"].tolist())
    return keys


def human_records(folder, engine, lost_last=0, lost_only=False):
    """Label the policy's turns in saved games. With `lost_last`, the last N policy turns of games the human won
    are "tail" positions: skipped normally, and the only ones kept when `lost_only` (label them with a long search)."""
    records, seen = [], set()
    for path in sorted(folder.rglob("*renju-*.json")):  # includes sub-folders (older games)
        data = json.loads(path.read_text(encoding="utf-8"))
        human = 1 if data["human_color"] == "black" else -1
        actions = [(m["row"] - 1) * SIZE + m["column"] - 1 for m in data["moves"]]
        board, histories = BatchBoard(1), np.full(CELLS, -1, dtype=np.int16)
        for ply, action in enumerate(actions):
            color = 1 if ply % 2 == 0 else -1
            cells = board.board[0]
            tail = bool(lost_last) and data.get("winner") == human and ply >= len(actions) - lost_last
            if color != human and ply >= 4 and tail == lost_only:
                key = canonical(cells, color)
                if key not in seen:
                    legal = rules.legal_mask(cells, color)
                    found = label_position(cells, color, actions[:ply], engine, legal)
                    if found is not None:
                        seen.add(key)
                        last = actions[ply - 1]
                        obs = np.stack((cells == color, cells == -color, np.arange(CELLS) == last,
                                        np.full(CELLS, color == 1))).astype(np.int8).reshape(4, SIZE, SIZE)
                        records.append((obs, found[0], legal.copy(), found[1], key, histories.copy()))
            histories[ply] = action
            board.place(np.array([action]), np.array([color], dtype=np.int8))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "eval", "human"))
    parser.add_argument("--models", type=Path, nargs="+")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--games", type=int, default=600, help="games for the training slice (test uses games//8)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--opp-ms", type=int, nargs="+", default=[60, 100])
    parser.add_argument("--label-ms", type=int, default=150)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--blunders", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--hunt", type=float, nargs="+", default=[0.0],
                        help="chance per game that the opponent plays any forced win (exact or deep VCF) it finds")
    parser.add_argument("--relabel-ms", type=int, default=0,
                        help="relabel the last engine-labelled positions of lost games with this search time")
    parser.add_argument("--lost-last", type=int, default=0,
                        help="human mode: tail length of games the human won (excluded, or the only part with --lost-only)")
    parser.add_argument("--lost-only", action="store_true")
    parser.add_argument("--prefix", default="rapfi")
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=880_000_001)
    args = parser.parse_args()
    reserved = keys_of(args.data, ("train", "test", "quiet_test", "onpolicy_test", "selfplay_test", "rapfi_test", "forbid_test", "rapfi2_test",
                                "deep_test", "forbid_black_test", "rapfi3_test", "rapfi4_test"))
    started = time.perf_counter()
    if args.mode == "eval":
        _, results = run(args.models, args.games, args.seed, args.workers, [0.0], 0.0, args.opp_ms, args.label_ms,
                         set(), label=False, blunders=args.blunders)
        print(json.dumps(summarize(results), indent=1), flush=True)
        return
    if args.mode == "human":
        engine = _engine(args.label_ms)
        try:
            records = human_records(args.folder, engine, args.lost_last, args.lost_only)
        finally:
            engine.close()
        records = [r for r in records if r[4] not in reserved]
        save(args.data / f"{args.prefix}_train.npz", records)
        print("human positions", len(records), {c: sum(r[3] == c for r in records) for c in range(1, 8)}, flush=True)
        return
    test_records, test_results = run(args.models, max(args.workers, args.games // 8), args.seed + 500, args.workers,
                                     args.betas, args.temperature, args.opp_ms, args.label_ms, reserved,
                                     blunders=args.blunders, hunts=args.hunt, relabel_ms=args.relabel_ms)
    save(args.data / f"{args.prefix}_test.npz", test_records)
    print("test", len(test_records), json.dumps(summarize(test_results)), f"{time.perf_counter() - started:.0f}s",
          flush=True)
    excluded = reserved | {r[4] for r in test_records}
    train_records, train_results = run(args.models, args.games, args.seed, args.workers, args.betas,
                                       args.temperature, args.opp_ms, args.label_ms, excluded, blunders=args.blunders,
                                       hunts=args.hunt, relabel_ms=args.relabel_ms)
    save(args.data / f"{args.prefix}_train.npz", train_records)
    print("train", len(train_records), {c: sum(r[3] == c for r in train_records) for c in range(1, 8)},
          json.dumps(summarize(train_results)), f"{time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
