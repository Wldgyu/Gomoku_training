"""Quiet opening/middle-game positions labelled by the Rapfi search engine (12x12, freestyle).

Positions come from sampled self-play of a candidate policy (diverse, plausible shapes) and are kept only when the
exact tactical teacher has no label (category 0) and nobody has an immediate winning cell. The label is the engine's
200 ms recommendation: a search suggestion, not a forced-win certificate. Category 7 marks these records.
"""
from pathlib import Path
import argparse
import json
import multiprocessing as mp

import numpy as np
import torch

from .defense_data import obs_key, save
from .env import BatchBoard, force_black_center, opponent_actions
from .midgame_data import canonical
from .models import load_policy
from .multistep_tactics import multi_targets, winning_cells

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "external/rapfi_250615/engine"
QUIET = 7


@torch.no_grad()
def selfplay_positions(model_path, games, seed, quota, reserved, plies=(4, 40), keep=0.2, temperature=1.0,
                       tactical_mix=0.15, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model, _ = load_policy(Path(model_path), device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    board = BatchBoard(games, 12)
    histories = np.full((games, 144), -1, dtype=np.int16)
    hidden = model.initial(games, device)
    states = {1: hidden.clone(), -1: hidden.clone()}
    zeros = torch.zeros(games, dtype=torch.bool, device=device)
    matrix = model.matrix()
    seen, items = set(reserved), []
    while len(items) < quota:
        colors = np.where(board.moves % 2 == 0, 1, -1).astype(np.int8)
        obs_np = board.observe(colors)
        legal = torch.from_numpy(board.legal()).to(device)
        mine = torch.from_numpy(colors == 1).to(device)[:, None]
        state = torch.where(mine, states[1], states[-1])
        logits, _, new_state = model.step(torch.from_numpy(obs_np).to(device), state, zeros, matrix)
        states[1] = torch.where(mine, new_state, states[1])
        states[-1] = torch.where(mine, states[-1], new_state)
        probs = torch.softmax(logits.masked_fill(~legal, -1e9) / temperature, 1)
        actions = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()
        tactical = opponent_actions(board, colors, rng, "tactical")
        actions = np.where(rng.random(games) < tactical_mix, tactical, actions)
        actions = force_black_center(board, colors, actions)
        eligible = (board.moves >= plies[0]) & (board.moves <= plies[1]) & ~board.finished & (rng.random(games) < keep)
        if eligible.any():
            labels, category = multi_targets(board, colors)
            for row in np.flatnonzero(eligible & (category == 0)):
                color = int(colors[row])
                if winning_cells(board.board[row], color).any() or winning_cells(board.board[row], -color).any():
                    continue
                key = canonical(board.board[row], color)
                if key in seen:
                    continue
                seen.add(key)
                items.append((obs_np[row].astype(np.int8), np.zeros(144, dtype=bool), QUIET, key, histories[row].copy()))
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
    from .external_engine import PiskvorkEngine
    if _engine is None:
        _engine = PiskvorkEngine([ENGINE / "pbrain-rapfi-windows-sse.exe"], size=12, turn_ms=task[2], cwd=ENGINE)
    obs, history = task[0], task[1]
    color = 1 if obs[3, 0, 0] else -1
    cells = ((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8)
    response = _engine.recommend(cells, color, history=history)
    return response["action"], response["evaluation"], response["seconds"]


def build(path, model_path, quota, seed, reserved, workers, turn_ms):
    if path.exists():
        with np.load(path) as saved:
            return set(saved["keys"].tolist())
    items = selfplay_positions(model_path, 128, seed, quota, reserved)
    print(f"{path.name}: {len(items)} positions to label", flush=True)
    with mp.Pool(workers) as pool:
        answers = pool.map(_label, [(i[0], i[4], turn_ms) for i in items], chunksize=16)
    labelled = []
    for item, (action, evaluation, seconds) in zip(items, answers):
        labels = np.zeros(144, dtype=bool)
        labels[action] = True
        labelled.append((item[0], labels, QUIET, item[3], item[4]))
    save(path, labelled)
    path.with_suffix(".json").write_text(json.dumps(dict(
        teacher="Rapfi 250615 classical evaluation plus search", rule="freestyle", size=12, turn_ms=turn_ms,
        generator=str(model_path), seed=seed, records=len(labelled),
        evaluations=[a[1] for a in answers], seconds=[round(a[2], 3) for a in answers],
        scope="search suggestions on quiet self-play states, not forced-win certificates"), indent=1))
    print(f"{path.name}: labelled {len(labelled)}", flush=True)
    return {i[3] for i in labelled}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--turn-ms", type=int, default=200)
    args = parser.parse_args()
    reserved = set()
    for name in ("onpolicy_test.npz", "synthetic_test.npz", "attack_test.npz", "train.npz", "attack_train.npz"):
        path = args.out / name
        if path.exists():
            with np.load(path) as saved:
                reserved |= {obs_key(o) for o in saved["obs"]}
    test_keys = build(args.out / "quiet_test.npz", args.model, args.test, 950_000_001, reserved, args.workers, args.turn_ms)
    build(args.out / "quiet_train.npz", args.model, args.train, 960_000_001, reserved | test_keys, args.workers, args.turn_ms)


if __name__ == "__main__":
    main()
