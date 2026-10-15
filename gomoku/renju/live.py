"""Fixed-seed games of a trained 15x15 Renju policy against the rule opponent, plus tactical accuracy in play."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models import load_policy
from . import tactics
from .data import has_threat_window
from .env import LearnerEnv, force_black_center
from .rules import CELLS


@torch.no_grad()
def audit(model, games=256, seed=321_000_481, opponent="tactical", max_steps=116, device="cpu"):
    env = LearnerEnv(games, seed, opponent)
    colors = env.colors.copy()
    state = model.initial(games, torch.device(device))
    resets = torch.ones(games, dtype=torch.bool, device=device)
    matrix = model.matrix()
    finished = np.zeros(games, dtype=bool)
    outcome = np.zeros(games, dtype=np.int8)
    plies = np.zeros(games, dtype=np.int32)
    counters = {name: [0, 0] for name in tactics.NAMES.values()}
    for _ in range(max_steps):
        obs, legal = env.observe()
        logits, _, state = model.step(torch.from_numpy(obs).to(device), state, resets, matrix)
        action = logits.masked_fill(~torch.from_numpy(legal).to(device), -1e9).argmax(1).cpu().numpy()
        action = force_black_center(env.games, env.colors, action)
        threatened = has_threat_window(env.games) & ~finished & (env.games.moves >= 5)
        for row in np.flatnonzero(threatened):
            labels, category = tactics.analyze(env.games.board[row], int(env.colors[row]))
            if category:
                name = tactics.NAMES[category]
                counters[name][0] += 1
                counters[name][1] += int(labels[action[row]])
        _, done, info = env.step(action)
        for row, result, ply in zip(info["rows"], info["results"], info["plies"]):
            if not finished[row]:
                finished[row], outcome[row], plies[row] = True, int(result), int(ply)
        resets = torch.from_numpy(done).to(device)
        if finished.all():
            break
    assert finished.all(), "unfinished games"
    by_color = {}
    for color, name in ((1, "black"), (-1, "white")):
        mask = colors == color
        by_color[name] = dict(games=int(mask.sum()), wins=int((outcome[mask] == 1).sum()),
                              losses=int((outcome[mask] == -1).sum()), draws=int((outcome[mask] == 0).sum()))
    return dict(games=games, wins=int((outcome == 1).sum()), losses=int((outcome == -1).sum()),
                draws=int((outcome == 0).sum()), mean_plies=float(plies.mean()), by_color=by_color,
                tactical={k: dict(positions=v[0], accuracy=(v[1] / v[0] if v[0] else None)) for k, v in counters.items()},
                seed=seed, opponent=opponent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=321_000_481)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = {}
    for path in args.models:
        model, _ = load_policy(path, torch.device("cpu"))
        rows[path.stem] = audit(model, args.games, args.seed)
        print(path.stem, json.dumps(rows[path.stem]), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
