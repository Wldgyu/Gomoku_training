"""Collect missed offensive plans from states visited by a frozen policy."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .dagger import MAX_TURNS, FIELDS, save_records
from .env import BatchBoard, geometry, opponent_actions
from .midgame_data import attack, canonical
from .tactics import position_key


def board_from_cells(cells):
    board = BatchBoard(1, 12)
    board.board[0] = cells
    board.moves[0] = int(np.count_nonzero(cells))
    values = cells[board.lines]
    board.line_counts[0, 0] = (values == 1).sum(1)
    board.line_counts[0, 1] = (values == -1).sum(1)
    return board


def stable_rng(cells, color, action):
    payload = cells.tobytes() + bytes([color + 1]) + int(action).to_bytes(2, "little")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def quiet_candidates(cells, color, limit=32):
    lines, incidence = geometry(12)
    values = cells[lines]
    own = (values == color).sum(1)
    enemy = (values == -color).sum(1)
    legal = cells == 0
    # Prefer moves extending several unblocked own lines; retain urgent blocks.
    attack_score = (((own + 1) ** 3) * (enemy == 0)) @ incidence.T
    defend_score = (((enemy + 1) ** 2) * (own == 0)) @ incidence.T
    score = attack_score + 0.4 * defend_score
    score[~legal] = -np.inf
    candidates = np.flatnonzero(legal & (score >= 8))
    if len(candidates) > limit:
        candidates = candidates[np.argsort(score[candidates], kind="stable")[-limit:]]
    return candidates[np.argsort(-score[candidates], kind="stable")]


def offensive_plan(cells, color, candidate_limit=32):
    """Return <=3-own-move labels against a stable tactical reply."""
    horizon, labels = attack(cells, color)
    if horizon:
        return horizon, labels, {}
    labels = np.zeros(144, dtype=bool)
    witnesses = {}
    for action in quiet_candidates(cells, color, candidate_limit):
        board = board_from_cells(cells.copy())
        board.place(np.array([action]), color)
        if board.finished[0]:
            labels[action] = True
            witnesses[int(action)] = None
            continue
        reply = int(opponent_actions(
            board, -color, stable_rng(cells, color, int(action)), "tactical"
        )[0])
        board.place(np.array([reply]), -color)
        if board.finished[0]:
            continue
        remaining, follow = attack(board.board[0], color)
        if 0 < remaining <= 2:
            labels[action] = True
            witnesses[int(action)] = (reply, remaining, follow.copy())
    horizon = 0
    if witnesses:
        horizon = 1 + max(value[1] for value in witnesses.values() if value is not None)
    return horizon, labels, witnesses


def replay_board(history):
    board = BatchBoard(1, 12)
    for ply, action in enumerate(history[history >= 0]):
        board.place(np.array([action]), 1 if ply % 2 == 0 else -1)
    return board


def record(board, color, history, labels, category=5, round_index=0):
    return {
        "obs": board.observe(color)[0].astype(np.int8),
        "labels": labels.copy(),
        "category": int(category),
        "keys": position_key(board.board[0], color),
        "histories": history.copy(),
        "round": round_index,
    }


def rollout_plan(history, color, first_labels, witnesses, round_index=0):
    """Store the missed first move and its certified tactical continuation."""
    board = replay_board(history)
    records = [record(board, color, history, first_labels, 5, round_index)]
    action = int(np.flatnonzero(first_labels)[0])
    witness = witnesses.get(action)
    moves = history[history >= 0].tolist()
    board.place(np.array([action]), color)
    moves.append(action)
    if board.finished[0]:
        return records
    if witness is None:
        reply = int(opponent_actions(
            board, -color, stable_rng(board.board[0], color, action), "tactical"
        )[0])
    else:
        reply, _, _ = witness
    board.place(np.array([reply]), -color)
    moves.append(reply)
    for _ in range(2):
        horizon, labels = attack(board.board[0], color)
        if not horizon:
            break
        prefix = np.full(144, -1, dtype=np.int16)
        prefix[:len(moves)] = moves
        records.append(record(board, color, prefix, labels, {1: 1, 2: 3, 3: 5}[horizon],
                              round_index))
        action = int(np.flatnonzero(labels)[0])
        before = board.board[0].copy()
        board.place(np.array([action]), color)
        moves.append(action)
        if board.finished[0]:
            break
        reply = int(opponent_actions(
            board, -color, stable_rng(before, color, action), "tactical"
        )[0])
        board.place(np.array([reply]), -color)
        moves.append(reply)
        if board.finished[0]:
            break
    return records

@torch.no_grad()
def collect_from_starts(model, start_data, games, seed, device, excluded=(), round_index=0):
    """Collect missed attacks at midgame states the policy actually enters in training."""
    from .midgame_env import restore_prefix_state

    with np.load(start_data) as saved:
        obs_all = saved["obs"].copy()
        labels_all = saved["labels"].copy()
        histories_all = saved["histories"].copy()
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(obs_all))[:min(games, len(obs_all))]
    obs_np = obs_all[chosen]
    labels = labels_all[chosen]
    histories = histories_all[chosen]
    colors = np.where(obs_np[:, 3, 0, 0] > 0, 1, -1).astype(np.int8)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    count = len(chosen)
    resets = torch.ones(count, dtype=torch.bool, device=torch_device)
    state = restore_prefix_state(
        model, model.initial(count, torch_device), resets,
        torch.from_numpy(histories.astype(np.int64)).to(torch_device),
    )
    obs = torch.from_numpy(obs_np).float().to(torch_device)
    legal_np = (obs_np[:, 0] + obs_np[:, 1]).reshape(count, -1) == 0
    legal = torch.from_numpy(legal_np).to(torch_device)
    logits, _, _ = model.step(obs, state, torch.zeros_like(resets), model.matrix())
    actions = logits.masked_fill(~legal, -1e9).argmax(1).cpu().numpy()

    seen = set(excluded)
    records = []
    missed = 0
    for row in range(count):
        if labels[row, actions[row]]:
            continue
        missed += 1
        cells = ((obs_np[row, 0] - obs_np[row, 1]) * colors[row]).reshape(-1).astype(np.int8)
        key = canonical(cells, int(colors[row]))
        if key in seen:
            continue
        continuation = rollout_plan(
            histories[row], int(colors[row]), labels[row], {}, round_index
        )
        for item in continuation:
            item_color = 1 if item["obs"][3, 0, 0] else -1
            item_cells = ((item["obs"][0] - item["obs"][1]) * item_color).reshape(-1).astype(np.int8)
            item_key = canonical(item_cells, item_color)
            if item_key not in seen:
                seen.add(item_key)
                records.append(item)
    return records, {
        "games": count,
        "offensive_opportunities": count,
        "missed_first_moves": missed,
        "records": len(records),
        "collection_seed": seed,
        "round": round_index,
        "source": str(start_data),
    }

@torch.no_grad()
def collect(model, games, seed, device, excluded=(), round_index=0):
    rng = np.random.default_rng(seed)
    board = BatchBoard(games, 12)
    colors = np.where(np.arange(games) % 2 == 0, 1, -1).astype(np.int8)
    histories = np.full((games, 144), -1, dtype=np.int16)
    white = colors == -1
    opening = opponent_actions(board, -colors, rng, "tactical")
    histories[white, 0] = opening[white]
    board.place(opening, -colors, white)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    state = model.initial(games, torch_device)
    zeros = torch.zeros(games, dtype=torch.bool, device=torch_device)
    active = np.ones(games, dtype=bool)
    found = np.zeros(games, dtype=bool)
    pending = [None] * games
    matrix = model.matrix()
    analysed_positions = 0
    opportunities = 0

    for _ in range(MAX_TURNS):
        if not active.any():
            break
        obs_np = board.observe(colors)
        legal_np = board.legal()
        obs = torch.from_numpy(obs_np).to(torch_device)
        legal = torch.from_numpy(legal_np).to(torch_device)
        logits, _, next_state = model.step(obs, state, zeros, matrix)
        state = torch.where(torch.from_numpy(active).to(torch_device)[:, None], next_state, state)
        actions = logits.masked_fill(~legal, -1e9).argmax(1).cpu().numpy()

        for row in np.flatnonzero(active & ~found & (board.moves >= 6)):
            analysed_positions += 1
            horizon, labels, witnesses = offensive_plan(board.board[row], int(colors[row]))
            if horizon:
                opportunities += 1
                if not labels[actions[row]]:
                    pending[row] = rollout_plan(
                        histories[row], int(colors[row]), labels, witnesses, round_index
                    )
                    found[row] = True

        rows = np.flatnonzero(active)
        histories[rows, board.moves[rows]] = actions[rows]
        board.place(actions, colors, active)
        reply_active = active & ~board.finished
        replies = opponent_actions(board, -colors, rng, "tactical")
        rows = np.flatnonzero(reply_active)
        histories[rows, board.moves[rows]] = replies[rows]
        board.place(replies, -colors, reply_active)
        active &= ~board.finished
    if active.any():
        raise RuntimeError("Offense collection exceeded game horizon")

    seen = set(excluded)
    records = []
    selected_games = 0
    for rows in pending:
        accepted = []
        if rows:
            selected_games += 1
            for row in rows:
                cells, color = ((row["obs"][0] - row["obs"][1]) *
                                (1 if row["obs"][3, 0, 0] else -1)).reshape(-1), (
                                    1 if row["obs"][3, 0, 0] else -1)
                key = canonical(cells.astype(np.int8), color)
                if key not in seen:
                    seen.add(key)
                    accepted.append(row)
        records.extend(accepted)
    stats = {
        "games": games, "analysed_positions": analysed_positions,
        "offensive_opportunities": opportunities,
        "selected_games": selected_games, "records": len(records),
        "collection_seed": seed, "round": round_index,
    }
    return records, stats


def main():
    import argparse
    import json
    from .models import load_policy
    from .dagger import heldout_canonical_keys

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--holdouts", type=Path, nargs="*", default=[])
    parser.add_argument("--start-data", type=Path)
    args = parser.parse_args()
    model, _ = load_policy(args.checkpoint, torch.device(args.device))
    collector = collect_from_starts if args.start_data else collect
    collector_args = (model, args.start_data, args.games, args.seed, args.device) \
        if args.start_data else (model, args.games, args.seed, args.device)
    records, stats = collector(
        *collector_args, excluded=heldout_canonical_keys(args.holdouts), round_index=0
    )
    save_records(args.output, records)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()






