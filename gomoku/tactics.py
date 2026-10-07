"""One-ply tactical labels from reachable, nonterminal freestyle positions.

Labels are a training-only auxiliary target, never an evaluation action override.
Win targets may contain multiple winning moves; block targets have exactly one
opponent winning cell and no immediate win for the player to move.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .env import BatchBoard, opponent_actions
from .models import masked_distribution


def targets(board: BatchBoard, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(board.count)
    slots = (colors == -1).astype(np.int64)
    own = board.line_counts[rows, slots]
    other = board.line_counts[rows, 1 - slots]
    wins = (((own == 4) & (other == 0)).astype(np.float32) @ board.incidence.T > 0) & board.legal()
    blocks = (((other == 4) & (own == 0)).astype(np.float32) @ board.incidence.T > 0) & board.legal()
    category = np.where(wins.any(1), 1, np.where(blocks.sum(1) == 1, 2, 0)).astype(np.int8)
    labels = np.where((category == 1)[:, None], wins, blocks)
    category[board.finished] = 0
    return labels, category


def position_key(cells: np.ndarray, color: int) -> str:
    return hashlib.sha256(cells.tobytes() + bytes([color + 1])).hexdigest()


def generate(path: Path, per_category: int, seed: int, tactical_probability: float,
             excluded: set[str] | None = None, size: int = 12) -> dict:
    if path.exists():
        with np.load(path) as saved:
            return {"positions": len(saved["category"]), "keys": set(saved["keys"].tolist())}
    rng = np.random.default_rng(seed)
    board = BatchBoard(256, size)
    history = np.full((board.count, size * size), -1, dtype=np.int16)
    observations, labels, categories, keys, histories = [], [], [], [], []
    seen = set() if excluded is None else set(excluded)
    counts = {1: 0, 2: 0}
    for iteration in range(100_000):
        colors = np.where(board.moves % 2 == 0, 1, -1).astype(np.int8)
        label, category = targets(board, colors)
        obs = board.observe(colors)
        for row in np.flatnonzero(category):
            cat = int(category[row])
            if counts[cat] >= per_category:
                continue
            key = position_key(board.board[row], int(colors[row]))
            if key in seen:
                continue
            seen.add(key)
            observations.append(obs[row].astype(np.int8))
            labels.append(label[row])
            categories.append(cat)
            keys.append(key)
            histories.append(history[row].copy())
            counts[cat] += 1
        if min(counts.values()) >= per_category:
            break
        random_actions = opponent_actions(board, colors, rng, "random")
        tactical_actions = opponent_actions(board, colors, rng, "tactical")
        actions = np.where(rng.random(board.count) < tactical_probability, tactical_actions, random_actions)
        history[np.arange(board.count), board.moves] = actions
        board.place(actions, colors)
        done = np.flatnonzero(board.finished)
        board.reset(done)
        history[done] = -1
    else:
        raise RuntimeError(f"Could not collect balanced tactical positions: {counts}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=np.array(observations), labels=np.array(labels),
                        category=np.array(categories, dtype=np.int8), keys=np.array(keys),
                        histories=np.array(histories), seed=seed, size=size,
                        tactical_probability=tactical_probability)
    print(f"dataset {path.name}: {counts}, simulation steps={iteration}", flush=True)
    return {"positions": len(keys), "keys": set(keys)}


class TacticalData:
    def __init__(self, path: Path, device: torch.device):
        with np.load(path) as saved:
            self.obs = torch.from_numpy(saved["obs"].astype(np.float32)).to(device)
            self.labels = torch.from_numpy(saved["labels"]).to(device)
            self.category = saved["category"].copy()
            self.histories = saved["histories"].copy()
        self.legal = (self.obs[:, 0] + self.obs[:, 1]).flatten(1) == 0

    def loss(self, model, batch: int) -> torch.Tensor:
        ids = torch.randint(len(self.obs), (batch,), device=self.obs.device)
        logits, _, _ = model.step(self.obs[ids], model.initial(batch, self.obs.device),
                                  torch.ones(batch, dtype=torch.bool, device=self.obs.device))
        log_probs = masked_distribution(logits, self.legal[ids]).logits
        # Maximize total probability of ANY correct immediate winning move.
        return -torch.logsumexp(log_probs.masked_fill(~self.labels[ids], -torch.inf), dim=1).mean()

    @torch.no_grad()
    def evaluate(self, model, history: bool = False) -> dict:
        success = []
        for start in range(0, len(self.obs), 128):
            obs = self.obs[start:start + 128]
            count = len(obs)
            device = obs.device
            state = model.initial(count, device)
            matrix = model.matrix()
            if history:
                board = BatchBoard(count, model.size)
                moves = self.histories[start:start + count]
                lengths = (moves >= 0).sum(1)
                colors = np.where(lengths % 2 == 0, 1, -1)
                # Reproduce observations at this player's actual decision times.
                for t in range(int(lengths.max())):
                    active = t < lengths
                    color = 1 if t % 2 == 0 else -1
                    observed = active & (colors == color)
                    logits, _, new = model.step(torch.from_numpy(board.observe(colors)).to(device),
                                                state, torch.zeros(count, dtype=torch.bool, device=device), matrix)
                    state = torch.where(torch.from_numpy(observed).to(device)[:, None], new, state)
                    board.place(np.where(active, moves[:, t], 0), color, active)
                if not np.array_equal(board.observe(colors), obs.cpu().numpy()):
                    raise RuntimeError("Tactical history reconstruction mismatch")
            logits, _, _ = model.step(obs, state, torch.zeros(count, dtype=torch.bool, device=device), matrix)
            actions = logits.masked_fill(~self.legal[start:start + count], -1e9).argmax(1)
            hit = self.labels[start:start + count].gather(1, actions[:, None]).squeeze(1)
            success.extend(hit.cpu().tolist())
        success = np.asarray(success)
        return {name: {"positions": int((self.category == cat).sum()),
                       "correct": int(success[self.category == cat].sum()),
                       "accuracy": float(success[self.category == cat].mean())}
                for name, cat in (("win", 1), ("block", 2))} | {"history_replayed": history}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    test = generate(a.output / "test.npz", 1024, 82_000_012, 0.8)
    validation = generate(a.output / "validation.npz", 512, 81_000_012, 0.3, test["keys"])
    generate(a.output / "train.npz", 10000, 80_000_012, 0.3, test["keys"] | validation["keys"])


if __name__ == "__main__":
    main()
