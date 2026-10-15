"""Head-to-head 15x15 Renju games between policies (both colors, greedy argmax, shared random openings)."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from ..models import load_policy
from .env import BatchBoard
from .rules import CELLS, CENTER, SIZE

OPENING_PLIES = 4


def make_openings(count, seed):
    """Black center, then white / black / white stones close to the center (never forbidden at this size)."""
    rng = np.random.default_rng(seed)
    openings = []
    near = [r * SIZE + c for r in range(5, 10) for c in range(5, 10) if r * SIZE + c != CENTER]
    while len(openings) < count:
        stones = [int(x) for x in rng.choice(near, OPENING_PLIES - 1, replace=False)]
        opening = tuple([CENTER] + stones)
        if opening not in openings:
            openings.append(opening)
    return openings


@torch.no_grad()
def play(model_a, model_b, openings, device="cpu"):
    """Result per game from A's point of view: rows [0:n] A is black, rows [n:2n] A is white."""
    n = len(openings)
    board = BatchBoard(2 * n)
    a_black = np.arange(2 * n) < n
    forced = torch.tensor(np.array(openings + openings), dtype=torch.long)
    dev = torch.device(device)
    models = (model_a.to(dev).eval(), model_b.to(dev).eval())
    states = [m.initial(2 * n, dev) for m in models]
    matrices = [m.matrix() for m in models]
    zeros = torch.zeros(2 * n, dtype=torch.bool, device=dev)
    for ply in range(CELLS):
        if board.finished.all():
            break
        colors = board.to_move()
        a_turn = (colors == 1) == a_black
        obs = torch.from_numpy(board.observe(colors)).to(dev)
        legal = torch.from_numpy(board.legal(colors)).to(dev)
        choice = []
        for k, model in enumerate(models):
            logits, _, new = model.step(obs, states[k], zeros, matrices[k])
            mover = torch.from_numpy(a_turn if k == 0 else ~a_turn).to(dev)[:, None]
            states[k] = torch.where(mover, new, states[k])
            choice.append(logits.masked_fill(~legal, -1e9).argmax(1).cpu().numpy())
        action = np.where(a_turn, choice[0], choice[1])
        if ply < OPENING_PLIES:
            action = forced[:, ply].numpy()
        board.place(action, colors, ~board.finished)
    winner = board.winner.astype(np.int64)
    result = np.where(a_black, winner, -winner)
    return result[:n], result[n:], board.moves


def summarize(result_as_black, result_as_white):
    f = lambda r: dict(wins=int((r == 1).sum()), draws=int((r == 0).sum()), losses=int((r == -1).sum()))
    both = np.concatenate([result_as_black, result_as_white])
    return dict(as_black=f(result_as_black), as_white=f(result_as_white), total=f(both),
                score=float(((both == 1).sum() + 0.5 * (both == 0).sum()) / len(both)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--openings", type=int, default=128)
    parser.add_argument("--seed", type=int, default=331_000_481)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    loaded = {p.stem: load_policy(p, torch.device("cpu"))[0] for p in args.models}
    openings = make_openings(args.openings, args.seed)
    table, totals = {}, {name: [0, 0, 0] for name in loaded}
    for a, b in itertools.combinations(loaded, 2):
        black, white, _ = play(loaded[a], loaded[b], openings, args.device)
        table[f"{a} vs {b}"] = summarize(black, white)
        both = np.concatenate([black, white])
        for name, sign in ((a, 1), (b, -1)):
            totals[name][0] += int((both == sign).sum())
            totals[name][1] += int((both == 0).sum())
            totals[name][2] += int((both == -sign).sum())
        print(a, "vs", b, json.dumps(table[f"{a} vs {b}"]), flush=True)
    ranking = {n: dict(wins=w, draws=d, losses=l, score=(w + 0.5 * d) / max(1, w + d + l)) for n, (w, d, l) in totals.items()}
    print(json.dumps(ranking, indent=1), flush=True)
    if args.out:
        args.out.write_text(json.dumps(dict(pairs=table, ranking=ranking, seed=args.seed, openings=args.openings),
                                       indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
