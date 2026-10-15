"""Batched 15x15 Renju environment with the same interface as gomoku.env.BatchBoard.

Black's first move is the center. A black stone that makes an overline, 4-4 or 3-3 (and no exact five)
loses at once; the legal mask normally hides those cells from black. Draw = full board.
"""
from __future__ import annotations

import numpy as np

from . import rules
from .rules import CELLS, CENTER, INCIDENCE, SIZE, WINDOWS


def legal_row(board_row, color):
    legal = board_row == 0
    if color == 1:
        allowed = legal & ~rules.forbidden_mask(board_row)
        if allowed.any():
            return allowed
    return legal


class BatchBoard:
    size = SIZE

    def __init__(self, count: int, size: int = SIZE):
        assert size == SIZE
        self.count = count
        self.board = np.zeros((count, CELLS), dtype=np.int8)
        self.moves = np.zeros(count, dtype=np.int32)
        self.last = np.full(count, -1, dtype=np.int64)
        self.finished = np.zeros(count, dtype=bool)
        self.winner = np.zeros(count, dtype=np.int8)
        self.forbidden_loss = np.zeros(count, dtype=bool)

    def reset(self, rows=None):
        rows = np.arange(self.count) if rows is None else np.asarray(rows)
        self.board[rows] = 0
        self.moves[rows] = 0
        self.last[rows] = -1
        self.finished[rows] = False
        self.winner[rows] = 0
        self.forbidden_loss[rows] = False

    def to_move(self):
        return np.where(self.moves % 2 == 0, 1, -1).astype(np.int8)

    def place(self, actions, colors, active=None):
        actions = np.asarray(actions, dtype=np.int64)
        colors = np.broadcast_to(np.asarray(colors, dtype=np.int8), (self.count,))
        active = ~self.finished if active is None else np.asarray(active, dtype=bool)
        rows = np.flatnonzero(active)
        if np.any(self.finished[rows]):
            raise ValueError("Cannot play on a finished board")
        chosen = actions[rows]
        if np.any((chosen < 0) | (chosen >= CELLS)):
            raise ValueError("Action is outside the board")
        if np.any(self.board[rows, chosen] != 0):
            raise ValueError("Cannot place a stone on an occupied intersection")
        self.board[rows, chosen] = colors[rows]
        self.last[rows] = chosen
        self.moves[rows] += 1
        for row, cell in zip(rows.tolist(), chosen.tolist()):
            color = int(colors[row])
            outcome = rules.move_result(self.board[row], cell, color)
            if outcome == "win":
                self.winner[row], self.finished[row] = color, True
            elif outcome == "forbidden":
                self.winner[row], self.finished[row], self.forbidden_loss[row] = -1, True, True
            elif self.moves[row] == CELLS:
                self.finished[row] = True
        return self.finished.copy()

    def observe(self, colors):
        colors = np.broadcast_to(np.asarray(colors), (self.count,))
        mine = self.board == colors[:, None]
        other = self.board == -colors[:, None]
        last = np.zeros_like(self.board, dtype=np.float32)
        valid = self.last >= 0
        last[np.flatnonzero(valid), self.last[valid]] = 1
        black = np.broadcast_to((colors == 1)[:, None], self.board.shape)
        return np.stack((mine, other, last, black), axis=1).astype(np.float32).reshape(self.count, 4, SIZE, SIZE)

    def legal(self, colors=None):
        colors = self.to_move() if colors is None else np.broadcast_to(np.asarray(colors), (self.count,))
        legal = self.board == 0
        for row in np.flatnonzero((colors == 1) & ~self.finished):
            legal[row] = legal_row(self.board[row], 1)
        return legal


def force_black_center(board, colors, actions):
    actions = np.array(actions, copy=True)
    colors = np.broadcast_to(np.asarray(colors), (board.count,))
    actions[(board.moves == 0) & (colors == 1) & ~board.finished] = CENTER
    return actions


def opponent_actions(board, colors, rng, kind="tactical"):
    """Random, or a shallow win/block/line-building rule player that respects black's forbidden cells."""
    colors = np.broadcast_to(np.asarray(colors), (board.count,))
    legal = board.legal(colors)
    scores = rng.random(legal.shape)
    if kind == "tactical":
        values = board.board[:, WINDOWS]
        own = (values == colors[:, None, None]).sum(2)
        other = (values == -colors[:, None, None]).sum(2)
        powers = np.array([0, 1, 6, 30, 2000, 0], dtype=np.float32)
        attack = powers[own] * (other == 0)
        defend = powers[other] * (own == 0)
        scores += (attack + 0.8 * defend) @ INCIDENCE
        wins = ((own == 4) & (other == 0)).astype(np.float32) @ INCIDENCE > 0
        blocks = ((other == 4) & (own == 0)).astype(np.float32) @ INCIDENCE > 0
        scores += wins * 100_000 + blocks * 10_000
    elif kind != "random":
        raise ValueError(kind)
    actions = np.where(legal, scores, -np.inf).argmax(axis=1)
    return force_black_center(board, colors, actions)


class LearnerEnv:
    """One learner decision per step against the rule opponent; learner color alternates every game."""

    def __init__(self, count=32, seed=1, opponent="tactical"):
        self.games = BatchBoard(count)
        self.rng = np.random.default_rng(seed)
        self.colors = np.where(np.arange(count) % 2 == 0, 1, -1).astype(np.int8)
        self.opponent = opponent
        self.reset(np.arange(count), flip=False)

    def reset(self, rows, flip=True):
        self.games.reset(rows)
        if flip:
            self.colors[rows] *= -1
        active = np.zeros(self.games.count, dtype=bool)
        active[rows] = self.colors[rows] == -1
        actions = opponent_actions(self.games, -self.colors, self.rng, self.opponent)
        self.games.place(actions, -self.colors, active)

    def observe(self):
        return self.games.observe(self.colors), self.games.legal(self.colors)

    def step(self, actions):
        actions = force_black_center(self.games, self.colors, actions)
        self.games.place(actions, self.colors)
        reply = ~self.games.finished
        opp = opponent_actions(self.games, -self.colors, self.rng, self.opponent)
        self.games.place(opp, -self.colors, reply)
        done = self.games.finished.copy()
        rewards = (self.games.winner * self.colors).astype(np.float32)
        rows = np.flatnonzero(done)
        info = dict(rows=rows, results=rewards[rows].copy(), plies=self.games.moves[rows].copy(),
                    colors=self.colors[rows].copy(), replies=opp, reply_active=reply)
        if len(rows):
            self.reset(rows)
        return rewards, done, info
