"""Exact forcing attack chains: attack, forced block, fork, block, win.

The teacher searches only chains that force a single blocking reply. It does
not claim to solve arbitrary Gomoku or quiet two-move plans. No search is used
by the policy during evaluation.
"""
from functools import lru_cache
import numpy as np
from .env import BatchBoard, geometry
from .expanded_tactics import expanded_targets

@lru_cache(maxsize=50000)
def features(cells_bytes, color, size):
    cells = np.frombuffer(cells_bytes, dtype=np.int8)
    lines, _ = geometry(size)
    values = cells[lines]
    own = (values == color).sum(1)
    other = (values == -color).sum(1)
    winning = np.zeros(size * size, dtype=bool)
    four = lines[(own == 4) & (other == 0)]
    if len(four):
        winning[four[cells[four] == 0]] = True
    graph = np.zeros((size * size, size * size), dtype=bool)
    three = lines[(own == 3) & (other == 0)]
    if len(three):
        empty = three[cells[three] == 0].reshape(-1, 2)
        graph[empty[:, 0], empty[:, 1]] = True
        graph[empty[:, 1], empty[:, 0]] = True
    return winning, graph.sum(1) >= 2, graph.any(1)

def winning_cells(cells, color, size=12):
    return features(cells.tobytes(), int(color), size)[0]

def safe_forks(cells, color, size=12):
    _, forks, _ = features(cells.tobytes(), int(color), size)
    enemy = winning_cells(cells, -color, size)
    if enemy.sum() == 1:
        return forks & enemy
    return forks if not enemy.any() else np.zeros_like(forks)

def forcing_chains(cells, color, size=12):
    """Return first-move -> (forced opponent block, own fork) witnesses."""
    wins, _, candidates = features(cells.tobytes(), int(color), size)
    if wins.any():
        return {}
    result = {}
    for action in np.flatnonzero(candidates & (cells == 0)):
        after = cells.copy()
        after[action] = color
        own = winning_cells(after, color, size)
        if own.sum() != 1 or winning_cells(after, -color, size).any():
            continue
        block = int(np.flatnonzero(own)[0])
        after[block] = -color
        follow = safe_forks(after, color, size)
        if follow.any():
            result[int(action)] = (block, int(np.flatnonzero(follow)[0]))
    return result

def safe_defenses(cells, color, size=12):
    """Exhaust all legal defenses against immediate/fork/forcing-chain wins."""
    safe = np.zeros(size * size, dtype=bool)
    for action in np.flatnonzero(cells == 0):
        after = cells.copy()
        after[action] = color
        # A move that wins immediately takes priority over the enemy plan.
        values = after[geometry(size)[0]]
        if ((values == color).sum(1) == 5).any():
            safe[action] = True
            continue
        if winning_cells(after, -color, size).any():
            continue
        if safe_forks(after, -color, size).any():
            continue
        if not forcing_chains(after, -color, size):
            safe[action] = True
    return safe

def multi_targets(board, colors):
    labels, category = expanded_targets(board, colors)
    for row in np.flatnonzero((category == 0) & ~board.finished):
        cells, color = board.board[row], int(colors[row])
        if winning_cells(cells, -color, board.size).any():
            labels[row] = False
            continue
        attack = forcing_chains(cells, color, board.size)
        if attack:
            labels[row] = False
            labels[row, list(attack)] = True
            category[row] = 5
        elif forcing_chains(cells, -color, board.size):
            safe = safe_defenses(cells, color, board.size)
            labels[row] = safe
            if safe.any():
                category[row] = 6
    return labels, category
