"""Replay fixed strong-opponent games and inspect multiple-threat failures."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from .env import LearnerEnv, opponent_actions
from .expanded_tactics import expanded_targets
from .history_aux import CATEGORIES
from .make_report import validate_replays
from .models import KINDS, load_policy
from .train import atomic_json


class RecordedEnv(LearnerEnv):
    def step(self, actions):
        self.games.place(actions, self.colors)
        opp = opponent_actions(self.games, -self.colors, self.rng, self.opponent)
        reply = ~self.games.finished
        self.games.place(opp, -self.colors, reply)
        done = self.games.finished.copy()
        rewards = (self.games.winner * self.colors).astype(np.float32)
        rows = np.flatnonzero(done)
        info = dict(rows=rows, results=rewards[rows].copy(), plies=self.games.moves[rows].copy(),
                    colors=self.colors[rows].copy(), replies=opp, reply_active=reply)
        if len(rows):
            self.reset(rows)
        return rewards, done, info


@torch.no_grad()
def audit(model, seed, games=256):
    env = RecordedEnv(games, seed=100_200_000 + seed, size=12, opponent="tactical")
    colors = env.colors.copy()
    histories = [[] if c == 1 else [int(env.games.last[i])] for i, c in enumerate(colors)]
    state = model.initial(games, torch.device("cpu"))
    resets = torch.ones(games, dtype=torch.bool)
    finished = np.zeros(games, dtype=bool)
    outcomes = np.zeros(games, dtype=np.int8)
    counters = {name: dict(positions=0, correct=0) for name in CATEGORIES.values()}
    loss_causes = {"missed_unique_block": 0, "already_multiple_winning_cells": 0, "other": 0}
    previous = np.zeros(games, dtype=np.int8)
    previous_correct = np.zeros(games, dtype=bool)
    prior_to_multiple = {"missed_prevent_fork": 0, "other_or_no_prevention_label": 0}
    matrix = model.matrix()
    replays = []
    for _ in range(74):
        obs, legal = env.observe()
        logits, _, state = model.step(torch.from_numpy(obs), state, resets, matrix)
        chosen = logits.masked_fill(~torch.from_numpy(legal), -1e9).argmax(1).numpy()
        label, category = expanded_targets(env.games, env.colors)
        correct = label[np.arange(games), chosen]
        for cat, name in CATEGORIES.items():
            active = (category == cat) & ~finished
            counters[name]["positions"] += int(active.sum())
            counters[name]["correct"] += int(correct[active].sum())
        slots = (env.colors == -1).astype(np.int64)
        own = env.games.line_counts[np.arange(games), slots]
        enemy = env.games.line_counts[np.arange(games), 1 - slots]
        winning_cells = (((enemy == 4) & (own == 0)).astype(np.float32) @ env.games.incidence.T > 0) & legal
        enemy_count = winning_cells.sum(1)
        active_rows = np.flatnonzero(~finished)
        for row in active_rows:
            histories[row].append(int(chosen[row]))
        _, done, info = env.step(chosen)
        for row in active_rows:
            if info["reply_active"][row]:
                histories[row].append(int(info["replies"][row]))
        for row, result in zip(info["rows"], info["results"]):
            if finished[row]:
                continue
            finished[row], outcomes[row] = True, int(result)
            if result == -1:
                if enemy_count[row] >= 2:
                    cause = "already_multiple_winning_cells"
                    prior = "missed_prevent_fork" if previous[row] == 4 and not previous_correct[row] else "other_or_no_prevention_label"
                    prior_to_multiple[prior] += 1
                elif category[row] == 2 and not correct[row]:
                    cause = "missed_unique_block"
                else:
                    cause = "other"
                loss_causes[cause] += 1
            if row < 8:
                replays.append(dict(size=12, a=model.kind, b="tactical", seed=seed,
                    a_color=int(colors[row]), winner=int(result * colors[row]), moves=histories[row],
                    opening_index=int(row), initial_opponent_move=bool(colors[row] == -1)))
        previous, previous_correct = category.copy(), correct.copy()
        resets = torch.from_numpy(done)
        if finished.all():
            break
    assert finished.all()
    validate_replays(replays)
    for row in counters.values():
        row["accuracy"] = row["correct"] / row["positions"] if row["positions"] else None
    return dict(kind=model.kind, seed=seed, games=games, wins=int((outcomes == 1).sum()),
        losses=int((outcomes == -1).sum()), draws=int((outcomes == 0).sum()),
        tactical_decisions=counters, terminal_loss_causes=loss_causes,
        preceding_turn_of_multiple_threat_losses=prior_to_multiple), replays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    primary = json.loads((a.input / "evaluation.json").read_text(encoding="utf-8"))
    rows, replays = [], []
    for seed in (4601, 4602, 4603):
        for kind in KINDS:
            model, _ = load_policy(a.input / f"{kind}_seed{seed}.pt", torch.device("cpu"))
            row, examples = audit(model, seed)
            original = next(r for r in primary["fixed_opponents"] if r["kind"] == kind
                            and r["seed"] == seed and r["stage"] == "final" and r["opponent"] == "tactical")
            assert all(row[k] == original[k] for k in ("wins", "draws", "losses", "games"))
            for name in ("win", "block"):
                assert row["tactical_decisions"][name] == original["tactical_decisions"][name]
            rows.append(row)
            replays.extend(examples)
            print(json.dumps(row), flush=True)
    atomic_json(a.input / "strong_match_audit.json", dict(runs=rows, same_primary_games_verified=True,
        interpretation="Terminal cause is the position before the last learner move, not a proof of earliest strategic error."))
    atomic_json(a.input / "strong_replays.json", dict(games=replays))


if __name__ == "__main__":
    main()
