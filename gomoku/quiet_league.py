"""Direct model-vs-model games: fixed argmax policies, no search, no teacher. 12x12 freestyle.

Each pairing plays every opening twice with colors swapped. An opening is four plies (black center, three random
nearby stones) that both models observe at their own turns, so recurrent states are built exactly as in live play.
"""
from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import torch

from .env import BatchBoard, black_center_action
from .models import load_policy

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/defense_probe_20261014"
KINDS = ("gru", "rnn", "rewired", "fly")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_openings(count, seed):
    rng = np.random.default_rng(seed)
    center = black_center_action(12)
    cr, cc = divmod(center, 12)
    seen, result = set(), []
    while len(result) < count:
        board = np.zeros(144, dtype=np.int8)
        board[center] = 1
        moves = [center]
        for ply, radius in ((1, 2), (2, 3), (3, 3)):
            cells = [r * 12 + c for r in range(max(0, cr - radius), min(12, cr + radius + 1))
                     for c in range(max(0, cc - radius), min(12, cc + radius + 1)) if board[r * 12 + c] == 0]
            action = int(rng.choice(cells))
            board[action] = 1 if ply % 2 == 0 else -1
            moves.append(action)
        key = tuple(moves)
        if key not in seen:
            seen.add(key)
            result.append(moves)
    return np.array(result, dtype=np.int64)


@torch.no_grad()
def play(model_a, model_b, openings):
    """Return per-game outcomes from A's perspective for A-black and A-white halves."""
    count = len(openings) * 2
    a_black = np.repeat([True, False], len(openings))
    opening = np.concatenate((openings, openings))
    board = BatchBoard(count, 12)
    zeros = torch.zeros(count, dtype=torch.bool, device=DEVICE)
    state_a, state_b = model_a.initial(count, DEVICE), model_b.initial(count, DEVICE)
    matrix_a, matrix_b = model_a.matrix(), model_b.matrix()
    plies = np.zeros(count, dtype=np.int32)
    for ply in range(144):
        if board.finished.all():
            break
        color = 1 if ply % 2 == 0 else -1
        a_moves = a_black == (color == 1)
        obs = torch.from_numpy(board.observe(np.full(count, color))).to(DEVICE)
        legal = torch.from_numpy(board.legal()).to(DEVICE)
        logits_a, _, new_a = model_a.step(obs, state_a, zeros, matrix_a)
        logits_b, _, new_b = model_b.step(obs, state_b, zeros, matrix_b)
        mask = torch.from_numpy(a_moves).to(DEVICE)[:, None]
        state_a = torch.where(mask, new_a, state_a)
        state_b = torch.where(mask, state_b, new_b)
        logits = torch.where(mask, logits_a, logits_b).masked_fill(~legal, -1e9)
        action = logits.argmax(1).cpu().numpy()
        if ply < opening.shape[1]:
            action = opening[:, ply]
        active = ~board.finished
        plies[active] = ply + 1
        board.place(action, color, active)
    winner = board.winner.astype(np.int64)
    a_color = np.where(a_black, 1, -1)
    outcome = winner * a_color  # +1 A wins, -1 B wins, 0 draw
    return outcome, a_black, plies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openings", type=int, default=128)
    parser.add_argument("--seed", type=int, default=331000481)
    parser.add_argument("--variant", default="atk_q", help="checkpoint suffix after scratch_<kind>_w64x6_")
    args = parser.parse_args()
    openings = make_openings(args.openings, args.seed)
    paths = {k: RUN / f"scratch_{k}_w64x6_{args.variant}.pt" for k in KINDS}
    models = {k: load_policy(p, DEVICE)[0] for k, p in paths.items()}
    table = {k: dict(win=0, draw=0, loss=0) for k in KINDS}
    pairs = []
    for a, b in itertools.combinations(KINDS, 2):
        outcome, a_black, plies = play(models[a], models[b], openings)
        row = dict(a=a, b=b, games=len(outcome), a_wins=int((outcome == 1).sum()), b_wins=int((outcome == -1).sum()),
                   draws=int((outcome == 0).sum()),
                   a_black=dict(wins=int(((outcome == 1) & a_black).sum()), losses=int(((outcome == -1) & a_black).sum()),
                                draws=int(((outcome == 0) & a_black).sum())),
                   a_white=dict(wins=int(((outcome == 1) & ~a_black).sum()), losses=int(((outcome == -1) & ~a_black).sum()),
                                draws=int(((outcome == 0) & ~a_black).sum())),
                   mean_plies=float(plies.mean()))
        pairs.append(row)
        table[a]["win"] += row["a_wins"]; table[a]["loss"] += row["b_wins"]; table[a]["draw"] += row["draws"]
        table[b]["win"] += row["b_wins"]; table[b]["loss"] += row["a_wins"]; table[b]["draw"] += row["draws"]
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for k in KINDS:
        games = table[k]["win"] + table[k]["draw"] + table[k]["loss"]
        table[k]["score"] = (table[k]["win"] + 0.5 * table[k]["draw"]) / games
    result = dict(variant=args.variant, openings=args.openings, seed=args.seed, games_per_pair=args.openings * 2,
                  pairs=pairs, table=table, checkpoints={k: str(p) for k, p in paths.items()},
                  note="fixed argmax policies; four-ply openings shared by both colors; fixed rule opponent not used")
    out = RUN / f"league_{args.variant}_{args.seed}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False))


if __name__ == "__main__":
    main()
