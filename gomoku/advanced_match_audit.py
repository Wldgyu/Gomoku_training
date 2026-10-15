"""Replay fixed strong-opponent games and inspect multiple-threat failures."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from .env import LearnerEnv, force_black_center, opponent_actions, black_center_action
from .expanded_tactics import expanded_targets
from .history_aux import CATEGORIES, burn_in
from .make_report import validate_replays
from .models import KINDS, load_policy
from .train import atomic_json


class RecordedEnv(LearnerEnv):
    def step(self, actions):
        if getattr(self, "force_center", True):
            actions = force_black_center(self.games, self.colors, actions)
        self.games.place(actions, self.colors)
        rr = np.arange(self.games.count)
        slots = (self.colors == -1).astype(np.int64)
        own, enemy = self.games.line_counts[rr, slots], self.games.line_counts[rr, 1-slots]
        legal = self.games.legal()
        own_wins = ((((own == 4) & (enemy == 0)).astype(np.float32) @ self.games.incidence.T > 0) & legal).sum(1)
        enemy_wins = ((((enemy == 4) & (own == 0)).astype(np.float32) @ self.games.incidence.T > 0) & legal).sum(1)
        safe_fork = (own_wins >= 2) & (enemy_wins == 0) & ~self.games.finished
        opp = opponent_actions(self.games, -self.colors, self.rng, self.opponent)
        reply = ~self.games.finished
        self.games.place(opp, -self.colors, reply)
        done = self.games.finished.copy()
        rewards = (self.games.winner * self.colors).astype(np.float32)
        rows = np.flatnonzero(done)
        info = dict(rows=rows, results=rewards[rows].copy(), plies=self.games.moves[rows].copy(),
                    colors=self.colors[rows].copy(), replies=opp, reply_active=reply, safe_fork=safe_fork)
        if len(rows):
            self.reset(rows)
        return rewards, done, info


def color_summary(colors, outcomes, color, color_block):
    positions, correct = color_block[color]
    mask = colors == color
    return dict(games=int(mask.sum()), wins=int(((outcomes == 1) & mask).sum()),
                losses=int(((outcomes == -1) & mask).sum()),
                draws=int(((outcomes == 0) & mask).sum()),
                defense_omissions=positions - correct, block_positions=positions)


@torch.no_grad()
def audit(model, seed, games=256, record_all=False, environment_seed=None,
          force_center=True, center_opponent_opening=False, opening_white_moves=None):
    env_seed = 100_200_000 + seed if environment_seed is None else environment_seed
    env = RecordedEnv(games, seed=env_seed, size=12, opponent="tactical")
    env.force_center = force_center
    if opening_white_moves is not None:
        opening_white_moves = np.asarray(opening_white_moves, dtype=np.int64)
        if (opening_white_moves.shape != (games,) or np.any(opening_white_moves == black_center_action(12))
                or np.any((opening_white_moves < 0) | (opening_white_moves >= 144))):
            raise ValueError("One legal non-center white opening per game is required")
        env.games.reset()
        env.games.place(np.full(games, black_center_action(12)), 1)
        env.games.place(opening_white_moves, -1)
        starting_histories = [[black_center_action(12), int(a)] for a in opening_white_moves]
        white = env.colors == -1
        reply = opponent_actions(env.games, 1, env.rng, env.opponent)
        env.games.place(reply, 1, white)
        for i in np.flatnonzero(white):
            starting_histories[i].append(int(reply[i]))
    elif center_opponent_opening:
        white_rows = np.flatnonzero(env.colors == -1)
        env.games.reset(white_rows)
        env.games.place(np.full(games, black_center_action(12)), 1, env.colors == -1)
    colors = env.colors.copy()
    histories = (starting_histories if opening_white_moves is not None
                 else [[] if c == 1 else [int(env.games.last[i])] for i, c in enumerate(colors)])
    if opening_white_moves is not None:
        prefixes = torch.full((games, 144), -1, dtype=torch.long)
        for i, history in enumerate(histories):
            prefixes[i, :len(history)] = torch.tensor(history)
        state = burn_in(model, prefixes)
        resets = torch.zeros(games, dtype=torch.bool)
    else:
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
    reached_multiple = np.zeros(games, dtype=bool)
    actual_forks = 0
    color_block = {1: [0, 0], -1: [0, 0]}
    black_first = []
    for _ in range(74):
        obs, legal = env.observe()
        logits, _, state = model.step(torch.from_numpy(obs), state, resets, matrix)
        chosen = logits.masked_fill(~torch.from_numpy(legal), -1e9).argmax(1).numpy()
        if force_center:
            chosen = force_black_center(env.games, env.colors, chosen)
        label, category = expanded_targets(env.games, env.colors)
        correct = label[np.arange(games), chosen]
        opening = (~finished) & (colors == 1) & (env.games.moves == 0)
        black_first.extend(int(action) for action in chosen[opening])
        block_rows = (category == 2) & ~finished
        for color in (1, -1):
            rows = block_rows & (colors == color)
            color_block[color][0] += int(rows.sum())
            color_block[color][1] += int(correct[rows].sum())
        for cat, name in CATEGORIES.items():
            active = (category == cat) & ~finished
            counters[name]["positions"] += int(active.sum())
            counters[name]["correct"] += int(correct[active].sum())
        slots = (env.colors == -1).astype(np.int64)
        own = env.games.line_counts[np.arange(games), slots]
        enemy = env.games.line_counts[np.arange(games), 1 - slots]
        winning_cells = (((enemy == 4) & (own == 0)).astype(np.float32) @ env.games.incidence.T > 0) & legal
        enemy_count = winning_cells.sum(1)
        own_count = (((((own == 4) & (enemy == 0)).astype(np.float32) @ env.games.incidence.T > 0) & legal)).sum(1)
        reached_multiple |= (enemy_count >= 2) & ~finished
        active_rows = np.flatnonzero(~finished)
        for row in active_rows:
            histories[row].append(int(chosen[row]))
        _, done, info = env.step(chosen)
        actual_forks += int((info["safe_fork"] & ~finished & (own_count < 2)).sum())
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
            if record_all or row < 8:
                replays.append(dict(size=12, a=model.kind, b="tactical", seed=seed,
                    a_color=int(colors[row]), winner=int(result * colors[row]), moves=histories[row],
                    opening_index=int(row), initial_opponent_move=bool(colors[row] == -1),
                    environment_seed=int(env_seed), loss_cause=cause if result == -1 else None))
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
        preceding_turn_of_multiple_threat_losses=prior_to_multiple,
        environment_seed=int(env_seed), actual_safe_forks_created=actual_forks,
        actual_safe_forks_per_game=actual_forks / games, games_reaching_multiple_threat=int(reached_multiple.sum()),
        multiple_threat_game_rate=float(reached_multiple.mean()),
        defense_omissions=counters["block"]["positions"] - counters["block"]["correct"],
        by_color={name: color_summary(colors, outcomes, color, color_block) for color, name in ((1, "black"), (-1, "white"))},
        black_first_moves=black_first, force_black_center=force_center,
        center_opponent_opening=center_opponent_opening,
        opening_white_moves=opening_white_moves.tolist() if opening_white_moves is not None else None), replays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--record-all", action="store_true")
    p.add_argument("--output-name", default="strong_replays.json")
    a = p.parse_args()
    torch.set_num_threads(4)
    primary = json.loads((a.input / "evaluation.json").read_text(encoding="utf-8"))
    rows, replays = [], []
    for seed in (4601, 4602, 4603):
        for kind in KINDS:
            model, _ = load_policy(a.input / f"{kind}_seed{seed}.pt", torch.device("cpu"))
            row, examples = audit(model, seed, record_all=a.record_all)
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
    atomic_json(a.input / a.output_name, dict(games=replays))


if __name__ == "__main__":
    main()
