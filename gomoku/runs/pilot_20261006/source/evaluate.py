"""Frozen-policy evaluation: fixed opponents and color-swapped opening pairs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .env import BatchBoard, LearnerEnv, opponent_actions
from .models import KINDS, Policy, load_policy, matched_hidden
from .train import DEFAULT_OUT, atomic_json


def result_counts(values: list[int] | np.ndarray) -> dict:
    values = np.asarray(values)
    win, draw, loss = (int(np.sum(values == x)) for x in (1, 0, -1))
    return dict(games=len(values), wins=win, draws=draw, losses=loss,
                win_rate=win / len(values), score=(win + 0.5 * draw) / len(values))


@torch.no_grad()
def versus_opponent(model: Policy, opponent: str, seed: int, games: int = 256) -> dict:
    device = next(model.parameters()).device
    env = LearnerEnv(games, seed=seed, opponent=opponent)
    state = model.initial(games, device)
    resets = torch.ones(games, dtype=torch.bool, device=device)
    finished = np.zeros(games, dtype=bool)
    outcome = np.zeros(games, dtype=np.int8)
    lengths = np.zeros(games, dtype=np.int64)
    original_colors = env.colors.copy()
    matrix = model.matrix()
    for _ in range(42):
        obs, legal = env.observe()
        logits, _, state = model.step(torch.from_numpy(obs).to(device), state, resets, matrix)
        actions = logits.masked_fill(~torch.from_numpy(legal).to(device), -1e9).argmax(1)
        _, done, info = env.step(actions.cpu().numpy())
        for row, value, plies in zip(info["rows"], info["results"], info["plies"]):
            if not finished[row]:
                finished[row] = True
                outcome[row], lengths[row] = int(value), int(plies)
        resets = torch.from_numpy(done).to(device)
        if finished.all():
            break
    if not finished.all():
        raise RuntimeError("A 9x9 game did not terminate within the board capacity")
    return dict(**result_counts(outcome), mean_plies=float(lengths.mean()),
                black=result_counts(outcome[original_colors == 1]),
                white=result_counts(outcome[original_colors == -1]), opponent=opponent,
                evaluation_seed=seed, policy="deterministic argmax on occupancy-masked logits")


@torch.no_grad()
def random_baseline(opponent: str, seed: int, games: int = 256) -> dict:
    env = LearnerEnv(games, seed=seed, opponent=opponent)
    rng = np.random.default_rng(seed + 1000)
    finished = np.zeros(games, dtype=bool)
    values = np.zeros(games, dtype=np.int8)
    for _ in range(42):
        actions = opponent_actions(env.games, env.colors, rng)
        _, _, info = env.step(actions)
        for row, value in zip(info["rows"], info["results"]):
            if not finished[row]:
                values[row], finished[row] = int(value), True
        if finished.all():
            return result_counts(values)
    raise RuntimeError("Random baseline game failed to terminate")


def opening_bank(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(81, 2, replace=False) for _ in range(count)])


@torch.no_grad()
def match(a: Policy, b: Policy, openings: np.ndarray, a_color: int) -> tuple[dict, list[dict]]:
    device = next(a.parameters()).device
    count = len(openings)
    board = BatchBoard(count)
    history = [list(map(int, opening)) for opening in openings]
    board.place(openings[:, 0], 1)
    board.place(openings[:, 1], -1)
    state_a, state_b = a.initial(count, device), b.initial(count, device)
    resets = torch.zeros(count, dtype=torch.bool, device=device)
    ma, mb = a.matrix(), b.matrix()
    color = 1
    for _ in range(79):
        active = ~board.finished
        if not active.any():
            break
        model, state, matrix = (a, state_a, ma) if color == a_color else (b, state_b, mb)
        obs = torch.from_numpy(board.observe(color)).to(device)
        logits, _, new_state = model.step(obs, state, resets, matrix)
        if color == a_color:
            state_a = new_state
        else:
            state_b = new_state
        legal = torch.from_numpy(board.legal()).to(device)
        actions = logits.masked_fill(~legal, -1e9).argmax(1).cpu().numpy()
        for row in np.flatnonzero(active):
            history[row].append(int(actions[row]))
        board.place(actions, color, active)
        color *= -1
    if not board.finished.all():
        raise RuntimeError("Head-to-head game did not terminate")
    values = board.winner * a_color
    examples = [dict(a_color=a_color, winner=int(board.winner[i]), moves=history[i],
                     opening_index=i) for i in range(min(8, count))]
    return dict(**result_counts(values), a_color=a_color, mean_plies=float(board.moves.mean())), examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4101, 4102, 4103])
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--openings", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.openings < 1:
        parser.error("Use a positive even fixed-opponent game count and positive opening count")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    models, fixed, fresh = {}, [], []
    for seed in args.seeds:
        for kind in KINDS:
            path = args.input / f"{kind}_seed{seed}.pt"
            model, saved = load_policy(path, device)
            if saved["completed_updates"] != saved["config"]["updates"]:
                raise RuntimeError("Final-budget checkpoint required")
            models[kind, seed] = model
            before = Policy(kind, seed, matched_hidden(kind), rewire_seed=seed % 5).to(device).eval()
            for opponent in ("random", "tactical"):
                eval_seed = 50_000_000 + seed + (0 if opponent == "random" else 100_000)
                row = dict(kind=kind, seed=seed, **versus_opponent(model, opponent, eval_seed, args.games))
                fixed.append(row)
                fresh.append(dict(kind=kind, seed=seed,
                                  **versus_opponent(before, opponent, eval_seed, args.games)))
                print(json.dumps(dict(event="fixed_opponent", **row)), flush=True)
            del before
    pairs, replays = [], []
    for seed in args.seeds:
        openings = opening_bank(args.openings, 60_000_000 + seed)
        atomic_json(args.input / f"openings_seed{seed}.json", {"moves": openings.tolist()})
        for ak, bk in itertools.combinations(KINDS, 2):
            colors = []
            for color in (1, -1):
                result, examples = match(models[ak, seed], models[bk, seed], openings, color)
                colors.append(result)
                replays.extend(dict(a=ak, b=bk, seed=seed, **example) for example in examples)
            total = dict(a=ak, b=bk, seed=seed, games=sum(r["games"] for r in colors),
                         wins=sum(r["wins"] for r in colors), draws=sum(r["draws"] for r in colors),
                         losses=sum(r["losses"] for r in colors), colors=colors,
                         mean_plies=float(np.mean([r["mean_plies"] for r in colors])))
            total["score"] = (total["wins"] + 0.5 * total["draws"]) / total["games"]
            pairs.append(total)
            print(json.dumps(dict(event="match", **total)), flush=True)
    result = dict(rule="9x9 freestyle: >=5 wins for both colors; no forbidden/opening rules",
                  observation="full board", evaluation_policy="frozen, deterministic argmax",
                  opponent_eval_games_per_run=args.games,
                  paired_two_ply_openings_per_seed=args.openings,
                  fixed_opponents=fixed, fresh_initialization=fresh, head_to_head=pairs,
                  random_baseline={o: random_baseline(o, 70_000_000, args.games)
                                   for o in ("random", "tactical")},
                  scope="same training seed pairs; cross-seed/cross-rewire league not yet included")
    atomic_json(args.input / "evaluation.json", result)
    atomic_json(args.input / "replays.json", {"games": replays})


if __name__ == "__main__":
    main()
