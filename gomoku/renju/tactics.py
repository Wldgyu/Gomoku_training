"""Exact tactical teacher for 15x15 Renju (labels are training targets, never evaluation overrides).

Categories (one position gets the first that applies):
  1 win                   an immediate five exists (black: exactly five)
  2 block                 the opponent has exactly one winning cell and we may legally occupy it
  3 create_fork           a legal move after which we own >= 2 distinct winning cells
  4 prevent_fork          the opponent has a fork move (and we have no forcing chain); every legal move after which it has none is labelled
  5 create_forcing_chain  a four that forces one reply, after which a fork exists (checked before 4: a forced win beats defending)
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


def _vcf_win(board, color, depth):
    """`color` (to move) wins by consecutive fours that each force one reply, at most `depth` fours.

    A four whose block cell is black-forbidden, or a move that creates two winning cells, ends the chain.
    Lines where the defender's reply makes its own four are skipped (never counted as wins).
    """
    if rules.winning_cells(board, color).any():
        return True
    if depth <= 0:
        return False
    for a, winners in rules.creates(board, color).items():
        after = board.copy()
        after[a] = color
        if rules.winning_cells(after, -color).any():
            continue
        if len(winners) >= 2:
            return True
        (b,) = winners
        if -color == 1 and rules.is_forbidden(rules.to_code(after, 1), b):
            return True
        after[b] = -color
        if rules.winning_cells(after, -color).any():
            continue
        if _vcf_win(after, color, depth - 1):
            return True
    return False


def deep_chains(board, color, depth=4):
    """First moves of a forced win by fours (VCF) of up to `depth` fours; empty when either side has a five."""
    if rules.winning_cells(board, color).any() or rules.winning_cells(board, -color).any():
        return []
    moves = []
    for a, winners in rules.creates(board, color).items():
        after = board.copy()
        after[a] = color
        if rules.winning_cells(after, -color).any():
            continue
        if len(winners) >= 2:
            moves.append(a)
            continue
        (b,) = winners
        if -color == 1 and rules.is_forbidden(rules.to_code(after, 1), b):
            moves.append(a)
            continue
        after[b] = -color
        if rules.winning_cells(after, -color).any():
            continue
        if _vcf_win(after, color, depth - 1):
            moves.append(a)
    return moves


def deep_defence(board, color, depth=4):
    """Legal moves after which the opponent has no fork move and no forced win by fours (VCF) of up to `depth`."""
    safe = np.zeros(CELLS, dtype=bool)
    after = board.copy()
    for move in np.flatnonzero(rules.legal_mask(board, color)):
        after[move] = color
        if not rules.forks(rules.creates(after, -color)) and not deep_chains(after, -color, depth):
            safe[move] = True
        after[move] = 0
    return safe


def analyze_deep(board, color, depth=4):
    """Like `analyze`, but a position without an exact verdict may hold a deep forced win / loss (VCF, <= depth fours).

    Returns (labels, category): 5 = attack (every first move of a forced win), 6 = defence (every legal move that
    leaves the opponent no forced win), else (none, 0). The exact verdict of `analyze` always wins when it exists.
    """
    labels, category = analyze(board, color)
    if category:
        return labels, category
    none = np.zeros(CELLS, dtype=bool)
    mine = deep_chains(board, color, depth)
    if mine:
        mask = none.copy()
        mask[mine] = True
        return mask, CREATE_CHAIN
    if deep_chains(board, -color, depth):
        safe = deep_defence(board, color, depth)
        if safe.any():
            return safe, PREVENT_CHAIN
    return none, 0


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
    chains = forcing_chains(board, color)
    if chains:  # a forced win by fours beats defending against a mere fork threat (opponent has no five threat now)
        mask = none.copy()
        mask[list(chains)] = True
        return mask, CREATE_CHAIN
    if rules.forks(rules.creates(board, -color)):
        safe = _safe_moves(board, color, False)
        if safe.any():
            return safe, PREVENT_FORK
    if forcing_chains(board, -color):
        safe = _safe_moves(board, color, True)
        if safe.any():
            return safe, PREVENT_CHAIN
    return none, 0
