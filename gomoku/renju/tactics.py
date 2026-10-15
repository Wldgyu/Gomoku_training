"""Exact tactical teacher for 15x15 Renju (labels are training targets, never evaluation overrides).

Categories (one position gets the first that applies):
  1 win                   an immediate five exists (black: exactly five)
  2 block                 the opponent has exactly one winning cell and we may legally occupy it
  3 create_fork           a legal move after which we own >= 2 distinct winning cells
  4 prevent_fork          the opponent has a fork move; every legal move after which it has none is labelled
  5 create_forcing_chain  a four that forces one reply, after which a fork exists
  6 prevent_forcing_chain the opponent has such a chain; every legal move that removes all of them is labelled
Black's forbidden cells are never offered as moves and can make a forced block impossible for black.
"""
from __future__ import annotations

import numpy as np

from . import rules
from .rules import CELLS

WIN, BLOCK, CREATE_FORK, PREVENT_FORK, CREATE_CHAIN, PREVENT_CHAIN = 1, 2, 3, 4, 5, 6
NAMES = {1: "win", 2: "block", 3: "create_fork", 4: "prevent_fork",
         5: "create_forcing_chain", 6: "prevent_forcing_chain"}


def forcing_chains(board, color):
    """{first move: (forced reply, follow-up fork)}; only fours that force exactly one reply are searched."""
    if rules.winning_cells(board, color).any():
        return {}
    chains = {}
    for a, winners in rules.creates(board, color).items():
        if len(winners) != 1:
            continue
        (b,) = winners
        after = board.copy()
        after[a] = color
        if rules.winning_cells(after, -color).any():
            continue
        if -color == 1 and rules.is_forbidden(rules.to_code(after, 1), b):
            chains[a] = (b, -1)  # black cannot legally block the four
            continue
        after[b] = -color
        if rules.winning_cells(after, -color).any():
            continue
        follow = rules.forks(rules.creates(after, color))
        if follow:
            chains[a] = (b, follow[0])
    return chains


def _safe_moves(board, color, need_chain_free):
    """Legal moves after which the opponent has no fork move (and optionally no forcing chain)."""
    safe = np.zeros(CELLS, dtype=bool)
    after = board.copy()
    for move in np.flatnonzero(rules.legal_mask(board, color)):
        after[move] = color
        if not rules.forks(rules.creates(after, -color)) and not (
                need_chain_free and forcing_chains(after, -color)):
            safe[move] = True
        after[move] = 0
    return safe


def analyze(board, color):
    """(labels bool[225], category) for the side `color` to move on a non-finished board."""
    none = np.zeros(CELLS, dtype=bool)
    wins = rules.winning_cells(board, color)
    if wins.any():
        return wins, WIN
    enemy_wins = rules.winning_cells(board, -color)
    if enemy_wins.any():
        if enemy_wins.sum() == 1:
            block = enemy_wins & rules.legal_mask(board, color)
            if block.any():
                return block, BLOCK
        return none, 0
    graph = rules.creates(board, color)
    forks = rules.forks(graph)
    if forks:
        mask = none.copy()
        mask[forks] = True
        return mask, CREATE_FORK
    if rules.forks(rules.creates(board, -color)):
        safe = _safe_moves(board, color, False)
        if safe.any():
            return safe, PREVENT_FORK
    chains = forcing_chains(board, color)
    if chains:
        mask = none.copy()
        mask[list(chains)] = True
        return mask, CREATE_CHAIN
    if forcing_chains(board, -color):
        safe = _safe_moves(board, color, True)
        if safe.any():
            return safe, PREVENT_CHAIN
    return none, 0
