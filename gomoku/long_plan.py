"""Bounded beam-search teacher for long offensive preparation in visited games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dagger import MAX_TURNS, heldout_canonical_keys, save_records
from .env import BatchBoard, geometry, opponent_actions
from .midgame_data import attack, canonical
from .models import load_policy
from .offense_dagger import board_from_cells, quiet_candidates, record, stable_rng


def has_win(cells, color):
    lines, _ = geometry(12)
    return bool(((cells[lines] == color).sum(1) == 5).any())


def tactical_reply(cells_before, cells_after, color, action):
    board = board_from_cells(cells_after.copy())
    reply = int(opponent_actions(
        board, -color, stable_rng(cells_before, color, int(action)), "tactical"
    )[0])
    board.place(np.array([reply]), -color)
    return reply, board.board[0].copy(), bool(board.finished[0] and board.winner[0] == -color)


def strategic_score(cells, color):
    lines, _ = geometry(12)
    values = cells[lines]
    own = (values == color).sum(1)
    enemy = (values == -color).sum(1)
    weights = np.array([0, 1, 8, 56, 500, 100000], dtype=np.float64)
    attack_score = (weights[own] * (enemy == 0)).sum()
    defense_score = (weights[enemy] * (own == 0)).sum()
    return float(attack_score - 0.85 * defense_score)


def search_plan(cells, color, max_preparations=4, candidate_limit=12, beam_width=48):
    """Find first moves leading to a <=3-move tactic after bounded preparation."""
    direct_horizon, direct_labels = attack(cells, color)
    if direct_horizon:
        return direct_labels, {int(a): dict(path=[], final_labels=direct_labels,
                                            total_own_moves=direct_horizon)
                               for a in np.flatnonzero(direct_labels)}, 1
    # entries: score, board after opponent reply, root move, alternating prep path
    beam = [(strategic_score(cells, color), cells.copy(), None, [])]
    successes = {}
    expanded_nodes = 0
    for depth in range(1, max_preparations + 1):
        expanded = []
        for _, state_cells, root, path in beam:
            for action in quiet_candidates(state_cells, color, candidate_limit):
                expanded_nodes += 1
                after = state_cells.copy()
                after[action] = color
                first = int(action) if root is None else root
                if has_win(after, color):
                    candidate = dict(path=path + [int(action)], final_labels=None,
                                     total_own_moves=depth)
                    if first not in successes or candidate["total_own_moves"] < successes[first]["total_own_moves"]:
                        successes[first] = candidate
                    continue
                reply, replied, opponent_won = tactical_reply(
                    state_cells, after, color, int(action)
                )
                if opponent_won:
                    continue
                next_path = path + [int(action), reply]
                horizon, labels = attack(replied, color)
                if horizon:
                    candidate = dict(path=next_path, final_labels=labels.copy(),
                                     total_own_moves=depth + horizon)
                    if first not in successes or candidate["total_own_moves"] < successes[first]["total_own_moves"]:
                        successes[first] = candidate
                    continue
                expanded.append((strategic_score(replied, color), replied, first, next_path))
        if successes:
            labels = np.zeros(144, dtype=bool)
            labels[list(successes)] = True
            return labels, successes, expanded_nodes
        # Deduplicate exact states and retain root diversity before filling by score.
        best = {}
        for item in expanded:
            key = (item[1].tobytes(), item[2])
            if key not in best or item[0] > best[key][0]:
                best[key] = item
        ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
        selected, used_roots = [], set()
        for item in ranked:
            if item[2] not in used_roots:
                selected.append(item)
                used_roots.add(item[2])
                if len(selected) == beam_width:
                    break
        if len(selected) < beam_width:
            chosen = {(x[1].tobytes(), x[2]) for x in selected}
            selected.extend(x for x in ranked if (x[1].tobytes(), x[2]) not in chosen) 
            selected = selected[:beam_width]
        beam = selected
        if not beam:
            break
    return np.zeros(144, dtype=bool), {}, expanded_nodes


def records_for_witness(history, color, root_labels, witness, round_index=0):
    """Convert every teacher move in one plan into history-conditioned supervision."""
    board = BatchBoard(1, 12)
    moves = history[history >= 0].tolist()
    for ply, action in enumerate(moves):
        board.place(np.array([action]), 1 if ply % 2 == 0 else -1)
    records = []
    path = witness["path"]
    for index in range(0, len(path), 2):
        own_action = path[index]
        labels = root_labels if index == 0 else np.eye(144, dtype=bool)[own_action]
        prefix = np.full(144, -1, dtype=np.int16)
        prefix[:len(moves)] = moves
        records.append(record(board, color, prefix, labels, 5, round_index))
        board.place(np.array([own_action]), color)
        moves.append(own_action)
        if board.finished[0] or index + 1 >= len(path):
            return records
        reply = path[index + 1]
        board.place(np.array([reply]), -color)
        moves.append(reply)
    labels = witness["final_labels"]
    for _ in range(3):
        if labels is None or not labels.any() or board.finished[0]:
            break
        prefix = np.full(144, -1, dtype=np.int16)
        prefix[:len(moves)] = moves
        horizon, current = attack(board.board[0], color)
        if not horizon:
            break
        records.append(record(board, color, prefix, current, {1: 1, 2: 3, 3: 5}[horizon],
                              round_index))
        action = int(np.flatnonzero(current)[0])
        before = board.board[0].copy()
        board.place(np.array([action]), color)
        moves.append(action)
        if board.finished[0]:
            break
        reply, replied, opponent_won = tactical_reply(before, board.board[0], color, action)
        board.place(np.array([reply]), -color)
        moves.append(reply)
        if opponent_won:
            break
        labels = attack(replied, color)[1]
    return records


@torch.no_grad()
def visited_losses(model, games, seed, device):
    rng = np.random.default_rng(seed)
    board = BatchBoard(games, 12)
    colors = np.where(np.arange(games) % 2 == 0, 1, -1).astype(np.int8)
    histories = np.full((games, 144), -1, dtype=np.int16)
    white = colors == -1
    opening = opponent_actions(board, -colors, rng, "tactical")
    histories[white, 0] = opening[white]
    board.place(opening, -colors, white)
    model = model.to(device)
    state = model.initial(games, torch.device(device))
    zeros = torch.zeros(games, dtype=torch.bool, device=device)
    active = np.ones(games, dtype=bool)
    snapshots = [[] for _ in range(games)]
    outcomes = np.zeros(games, dtype=np.int8)
    matrix = model.matrix()
    for _ in range(MAX_TURNS):
        if not active.any():
            break
        obs = torch.from_numpy(board.observe(colors)).to(device)
        legal = torch.from_numpy(board.legal()).to(device)
        logits, _, next_state = model.step(obs, state, zeros, matrix)
        state = torch.where(torch.from_numpy(active).to(device)[:, None], next_state, state)
        actions = logits.masked_fill(~legal, -1e9).argmax(1).cpu().numpy()
        for row in np.flatnonzero(active & (board.moves >= 8) & (board.moves <= 36)):
            # Search every second learner turn to cover early/middle/late construction.
            if ((int(board.moves[row]) - int(colors[row] == -1)) // 2) % 2 == 0:
                snapshots[row].append((board.board[row].copy(), histories[row].copy(), int(actions[row])))
        rows = np.flatnonzero(active)
        histories[rows, board.moves[rows]] = actions[rows]
        board.place(actions, colors, active)
        reply_active = active & ~board.finished
        replies = opponent_actions(board, -colors, rng, "tactical")
        rows = np.flatnonzero(reply_active)
        histories[rows, board.moves[rows]] = replies[rows]
        board.place(replies, -colors, reply_active)
        done = active & board.finished
        outcomes[done] = board.winner[done] * colors[done]
        active &= ~board.finished
    if active.any():
        raise RuntimeError("Visited game exceeded horizon")
    return colors, snapshots, outcomes


def collect(model, games, seed, device, excluded=(), max_preparations=4,
            candidate_limit=12, beam_width=48, round_index=0):
    colors, snapshots, outcomes = visited_losses(model, games, seed, device)
    records, seen = [], set(excluded)
    selected_games = plans_found = expanded_nodes = searched_states = 0
    model_misses = 0
    depth_counts = {}
    for game in range(games):
        if outcomes[game] != -1:
            continue
        selected = None
        for cells, history, chosen in snapshots[game]:
            searched_states += 1
            labels, witnesses, nodes = search_plan(
                cells, int(colors[game]), max_preparations, candidate_limit, beam_width
            )
            expanded_nodes += nodes
            if labels.any():
                plans_found += 1
                if not labels[chosen]:
                    model_misses += 1
                    first = min(witnesses, key=lambda a: (witnesses[a]["total_own_moves"], a))
                    selected = records_for_witness(
                        history, int(colors[game]), labels, witnesses[first], round_index
                    )
                    depth = witnesses[first]["total_own_moves"]
                    depth_counts[str(depth)] = depth_counts.get(str(depth), 0) + 1
                    break
        if selected:
            selected_games += 1
            for row in selected:
                obs = row["obs"]
                color = 1 if obs[3, 0, 0] else -1
                cells = ((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8)
                key = canonical(cells, color)
                if key not in seen:
                    seen.add(key)
                    records.append(row)
    stats = dict(games=games, losses=int((outcomes == -1).sum()), wins=int((outcomes == 1).sum()),
                 searched_states=searched_states, plans_found=plans_found,
                 missed_plans=model_misses, selected_games=selected_games, records=len(records),
                 expanded_nodes=expanded_nodes, total_own_move_depths=depth_counts,
                 max_preparations=max_preparations, candidate_limit=candidate_limit,
                 beam_width=beam_width, collection_seed=seed)
    return records, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-preparations", type=int, default=4)
    p.add_argument("--candidate-limit", type=int, default=12)
    p.add_argument("--beam-width", type=int, default=48)
    p.add_argument("--holdouts", type=Path, nargs="*", default=[])
    a = p.parse_args()
    model, _ = load_policy(a.checkpoint, torch.device(a.device))
    records, stats = collect(model, a.games, a.seed, a.device,
        heldout_canonical_keys(a.holdouts), a.max_preparations,
        a.candidate_limit, a.beam_width)
    save_records(a.output, records)
    a.output.with_suffix(".json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
