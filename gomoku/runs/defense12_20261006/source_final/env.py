"""Freestyle Gomoku. Both colors win with at least five contiguous stones.

No forbidden patterns or opening restrictions. Coordinates are zero based;
action = row * size + column. Board values: 0 empty, 1 black, -1 white.
The learning policy receives the full board and an occupancy-only action mask.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def geometry(size: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 5:
        raise ValueError("Board must be at least 5x5")
    lines = []
    for row in range(size):
        for col in range(size):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if 0 <= row + 4 * dr < size and 0 <= col + 4 * dc < size:
                    lines.append([(row + k * dr) * size + col + k * dc for k in range(5)])
    lines_array = np.array(lines, dtype=np.int64)
    incidence = np.zeros((size * size, len(lines)), dtype=np.int8)
    for i, line in enumerate(lines):
        incidence[line, i] = 1
    return lines_array, incidence


class BatchBoard:
    """Independent boards, incremental line counts, explicit terminal states."""

    def __init__(self, count: int, size: int = 9):
        self.count, self.size = count, size
        self.lines, self.incidence = geometry(size)
        self.board = np.zeros((count, size * size), dtype=np.int8)
        self.line_counts = np.zeros((count, 2, len(self.lines)), dtype=np.int8)
        self.moves = np.zeros(count, dtype=np.int32)
        self.last = np.full(count, -1, dtype=np.int64)
        self.finished = np.zeros(count, dtype=bool)
        self.winner = np.zeros(count, dtype=np.int8)

    def reset(self, rows: np.ndarray | None = None) -> None:
        rows = np.arange(self.count) if rows is None else np.asarray(rows)
        self.board[rows] = 0
        self.line_counts[rows] = 0
        self.moves[rows] = 0
        self.last[rows] = -1
        self.finished[rows] = False
        self.winner[rows] = 0

    def place(self, actions: np.ndarray, colors: np.ndarray | int,
              active: np.ndarray | None = None) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int64)
        colors = np.broadcast_to(np.asarray(colors, dtype=np.int8), (self.count,))
        active = ~self.finished if active is None else np.asarray(active, dtype=bool)
        if actions.shape != (self.count,) or active.shape != (self.count,):
            raise ValueError("One action/active flag per board is required")
        rows = np.flatnonzero(active)
        if np.any(self.finished[rows]):
            raise ValueError("Cannot play on a finished board")
        chosen = actions[rows]
        if np.any((chosen < 0) | (chosen >= self.size * self.size)):
            raise ValueError("Action is outside the board")
        if np.any(self.board[rows, chosen] != 0):
            raise ValueError("Cannot place a stone on an occupied intersection")
        if np.any(np.abs(colors[rows]) != 1):
            raise ValueError("Color must be black (+1) or white (-1)")
        self.board[rows, chosen] = colors[rows]
        self.last[rows] = chosen
        self.moves[rows] += 1
        slots = (colors[rows] == -1).astype(np.int64)
        self.line_counts[rows, slots] += self.incidence[chosen]
        won = self.line_counts[rows, slots].max(axis=1) >= 5
        self.winner[rows[won]] = colors[rows[won]]
        self.finished[rows] = won | (self.moves[rows] == self.size * self.size)
        return self.finished.copy()

    def observe(self, colors: np.ndarray | int) -> np.ndarray:
        colors = np.broadcast_to(np.asarray(colors), (self.count,))
        mine = self.board == colors[:, None]
        other = self.board == -colors[:, None]
        last = np.zeros_like(self.board, dtype=np.float32)
        valid = self.last >= 0
        last[np.flatnonzero(valid), self.last[valid]] = 1
        black = np.broadcast_to((colors == 1)[:, None], self.board.shape)
        return np.stack((mine, other, last, black), axis=1).astype(np.float32).reshape(
            self.count, 4, self.size, self.size)

    def legal(self) -> np.ndarray:
        return self.board == 0


def opponent_actions(board: BatchBoard, colors: np.ndarray | int,
                     rng: np.random.Generator, kind: str = "random") -> np.ndarray:
    """Random, or shallow win/block/line-building opponent; never a learner hint."""
    colors = np.broadcast_to(np.asarray(colors), (board.count,))
    legal = board.legal()
    # Continuous independent noise makes argmax exactly uniform over empty cells.
    scores = rng.random(legal.shape)
    if kind in ("tactical", "weak"):
        rows = np.arange(board.count)
        slots = (colors == -1).astype(np.int64)
        own, other = board.line_counts[rows, slots], board.line_counts[rows, 1 - slots]
        powers = np.array([0, 1, 6, 30, 2000, 0], dtype=np.float32)
        attack = powers[own] * (other == 0)
        defend = powers[other] * (own == 0)
        scores += (attack + 0.8 * defend) @ board.incidence.T
        wins = ((own == 4) & (other == 0)).astype(np.float32) @ board.incidence.T > 0
        blocks = ((other == 4) & (own == 0)).astype(np.float32) @ board.incidence.T > 0
        scores += wins * 100_000 + blocks * 10_000
        if kind == "weak":
            # Independently ignore the heuristic on 70% of opponent turns.
            random_turn = rng.random(board.count) < 0.7
            scores[random_turn] = rng.random(legal.shape)[random_turn]
    elif kind != "random":
        raise ValueError(kind)
    return np.where(legal, scores, -np.inf).argmax(axis=1)


class LearnerEnv:
    """One learning decision per step; opponent replies, then terminal rows reset.

Rewards are +1 for a win, -1 for a loss and 0 otherwise, including draws.
No tactical features, teacher moves, or intermediate reward shaping are supplied.
"""

    def __init__(self, count: int = 32, seed: int = 1, size: int = 9,
                 opponent: str = "random"):
        self.games = BatchBoard(count, size)
        self.rng = np.random.default_rng(seed)
        self.colors = np.where(np.arange(count) % 2 == 0, 1, -1).astype(np.int8)
        self.opponent = opponent
        self.reset(np.arange(count), flip=False)

    def reset(self, rows: np.ndarray, flip: bool = True) -> None:
        self.games.reset(rows)
        if flip:
            self.colors[rows] *= -1
        active = np.zeros(self.games.count, dtype=bool)
        active[rows] = self.colors[rows] == -1
        actions = opponent_actions(self.games, -self.colors, self.rng, self.opponent)
        self.games.place(actions, -self.colors, active)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        return self.games.observe(self.colors), self.games.legal()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        self.games.place(actions, self.colors)
        opp = opponent_actions(self.games, -self.colors, self.rng, self.opponent)
        self.games.place(opp, -self.colors, ~self.games.finished)
        done = self.games.finished.copy()
        rewards = (self.games.winner * self.colors).astype(np.float32)
        rows = np.flatnonzero(done)
        info = {"rows": rows, "results": rewards[rows].copy(),
                "plies": self.games.moves[rows].copy(), "colors": self.colors[rows].copy()}
        if len(rows):
            self.reset(rows)
        return rewards, done, info
