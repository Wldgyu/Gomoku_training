"""On-policy failure collection for selective DAgger-style supervision."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .env import BatchBoard, force_black_center, opponent_actions
from .midgame_data import canonical
from .multistep_tactics import multi_targets
from .tactics import position_key

FIELDS = ("obs", "labels", "category", "keys", "histories")
MAX_TURNS = 73


def heldout_canonical_keys(paths):
    """Reserve every history prefix and target board, including D4 symmetries."""
    keys = set()
    for path in paths:
        with np.load(path) as saved:
            histories = saved["histories"].copy()
            observations = saved["obs"].copy()
        for history, obs in zip(histories, observations):
            board = BatchBoard(1, 12)
            for ply, action in enumerate(history[history >= 0]):
                color = 1 if ply % 2 == 0 else -1
                keys.add(canonical(board.board[0], color))
                board.place(np.array([action]), color)
            color = 1 if obs[3, 0, 0] else -1
            cells = ((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8)
            keys.add(canonical(cells, color))
    return keys


def records_from_file(path):
    if not path.exists():
        return []
    with np.load(path) as saved:
        arrays = {field: saved[field].copy() for field in FIELDS}
        rounds = saved["round"].copy() if "round" in saved else np.zeros(len(arrays["keys"]), dtype=np.int8)
    return [
        {field: arrays[field][i].copy() if isinstance(arrays[field][i], np.ndarray)
         else arrays[field][i].item() for field in FIELDS} | {"round": int(rounds[i])}
        for i in range(len(arrays["keys"]))
    ]


def save_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {field: np.array([row[field] for row in records]) for field in FIELDS}
    arrays["round"] = np.array([row.get("round", 0) for row in records], dtype=np.int8)
    arrays["size"] = np.array(12)
    np.savez_compressed(path, **arrays)


@torch.no_grad()
def collect(model, games, seed, device, excluded=(), round_index=0,
            center_opponent_opening=False, skip_excluded_departures=False,
            opponent_kind="tactical"):
    """Collect the first certified unsafe learner action from each lost game."""
    rng = np.random.default_rng(seed)
    board = BatchBoard(games, 12)
    colors = np.where(np.arange(games) % 2 == 0, 1, -1).astype(np.int8)
    histories = np.full((games, 144), -1, dtype=np.int16)

    white = colors == -1
    opening = opponent_actions(board, -colors, rng, opponent_kind)
    if center_opponent_opening:
        opening = force_black_center(board, -colors, opening)
    histories[white, 0] = opening[white]
    board.place(opening, -colors, white)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    state = model.initial(games, torch_device)
    zeros = torch.zeros(games, dtype=torch.bool, device=torch_device)
    active = np.ones(games, dtype=bool)
    pending = [None] * games
    records = []
    seen = set(excluded)
    first_departures = 0
    losses = 0
    losses_without_certified_departure = 0
    matrix = model.matrix()

    for _ in range(MAX_TURNS):
        if not active.any():
            break
        obs_np = board.observe(colors)
        legal_np = board.legal()
        obs = torch.from_numpy(obs_np).to(torch_device)
        legal = torch.from_numpy(legal_np).to(torch_device)
        logits, _, next_state = model.step(obs, state, zeros, matrix)
        state = torch.where(torch.from_numpy(active).to(torch_device)[:, None], next_state, state)
        actions = force_black_center(board, colors, logits.masked_fill(~legal, -1e9).argmax(1).cpu().numpy())

        labels, categories = multi_targets(board, colors)
        for row in np.flatnonzero(active & (categories > 0)):
            if pending[row] is None and not labels[row, actions[row]]:
                if skip_excluded_departures and canonical(board.board[row], int(colors[row])) in seen:
                    continue
                pending[row] = {
                    "obs": obs_np[row].astype(np.int8),
                    "labels": labels[row].copy(),
                    "category": int(categories[row]),
                    "keys": position_key(board.board[row], int(colors[row])),
                    "histories": histories[row].copy(),
                    "round": round_index,
                    "_canonical": canonical(board.board[row], int(colors[row])),
                }
                first_departures += 1

        rows = np.flatnonzero(active)
        histories[rows, board.moves[rows]] = actions[rows]
        board.place(actions, colors, active)
        reply_active = active & ~board.finished
        replies = opponent_actions(board, -colors, rng, opponent_kind)
        rows = np.flatnonzero(reply_active)
        histories[rows, board.moves[rows]] = replies[rows]
        board.place(replies, -colors, reply_active)

        finished = active & board.finished
        for row in np.flatnonzero(finished):
            result = int(board.winner[row] * colors[row])
            if result == -1:
                losses += 1
                candidate = pending[row]
                if candidate is None:
                    losses_without_certified_departure += 1
                elif candidate["_canonical"] not in seen:
                    seen.add(candidate["_canonical"])
                    candidate.pop("_canonical")
                    records.append(candidate)
        active &= ~board.finished
    if active.any():
        raise RuntimeError("DAgger collection exceeded the 12x12 game horizon")
    stats = {
        "games": games,
        "losses": losses,
        "first_certified_departures": first_departures,
        "selected_loss_records": len(records),
        "losses_without_certified_departure": losses_without_certified_departure,
        "excluded_or_duplicate": losses - losses_without_certified_departure - len(records),
        "collection_seed": seed,
        "round": round_index,
        "skip_excluded_departures": skip_excluded_departures,
        "opponent_kind": opponent_kind,
    }
    return records, stats


def merge(path, new_records):
    records = records_from_file(path)
    known = {row["keys"] for row in records}
    records.extend(row for row in new_records if row["keys"] not in known)
    save_records(path, records)
    return records


def record_canonical_keys(path):
    keys = set()
    for row in records_from_file(path):
        obs = row["obs"]
        color = 1 if obs[3, 0, 0] else -1
        cells = ((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8)
        keys.add(canonical(cells, color))
    return keys


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None



