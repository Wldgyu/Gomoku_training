"""Exact 15x15 Renju rules.

Black moves first (forced to the center in this project) and may not play an overline (six or more in a
row), a double four (4-4) or a double open three (3-3). A move that makes exactly five wins for black even
when it would otherwise be forbidden. White is unrestricted and wins with five or more. Board arrays are
int8 with 0 empty, 1 black, -1 white; action = row * 15 + column.

Line analysis works on short byte strings (13 cells through a stone, code 0 empty, 1 mine, 2 other,
3 off-board) and is cached, so repeated shapes cost almost nothing.
"""
from functools import lru_cache

import numpy as np

SIZE = 15
CELLS = SIZE * SIZE
CENTER = (SIZE // 2) * SIZE + SIZE // 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
RADIUS = 6
MID = RADIUS
MAX_DEPTH = 3


def _build():
    line = np.full((4, CELLS, 2 * RADIUS + 1), CELLS, dtype=np.int64)
    windows, window_dirs = [], []
    for d, (dr, dc) in enumerate(DIRECTIONS):
        for r in range(SIZE):
            for c in range(SIZE):
                for k in range(-RADIUS, RADIUS + 1):
                    rr, cc = r + k * dr, c + k * dc
                    if 0 <= rr < SIZE and 0 <= cc < SIZE:
                        line[d, r * SIZE + c, k + RADIUS] = rr * SIZE + cc
                if 0 <= r + 4 * dr < SIZE and 0 <= c + 4 * dc < SIZE:
                    windows.append([(r + k * dr) * SIZE + c + k * dc for k in range(5)])
                    window_dirs.append(d)
    return line, np.array(windows, dtype=np.int64), np.array(window_dirs, dtype=np.int64)


LINE, WINDOWS, WINDOW_DIRS = _build()
INCIDENCE = np.zeros((len(WINDOWS), CELLS), dtype=np.float32)
for _i, _w in enumerate(WINDOWS):
    INCIDENCE[_i, _w] = 1


def to_code(board, color):
    """uint8 array of CELLS+1 entries: 0 empty, 1 stones of `color`, 2 other stones, 3 off-board sentinel."""
    code = np.empty(CELLS + 1, dtype=np.uint8)
    code[:CELLS] = np.where(board == color, 1, np.where(board == 0, 0, 2))
    code[CELLS] = 3
    return code


# ---------------------------------------------------------------- cached line analysis
@lru_cache(maxsize=None)
def bounds(s, i):
    a = i
    while a > 0 and s[a - 1] == 1:
        a -= 1
    b = i
    while b < len(s) - 1 and s[b + 1] == 1:
        b += 1
    return a, b


@lru_cache(maxsize=1 << 20)
def completions(s, black, include=None):
    """Empty indexes where one more own stone makes a (black: exact) five, optionally containing `include`."""
    found = []
    for j, value in enumerate(s):
        if value != 0:
            continue
        t = s[:j] + b"\x01" + s[j + 1:]
        a, b = bounds(t, j)
        run = b - a + 1
        if (run == 5 if black else run >= 5) and (include is None or a <= include <= b):
            found.append(j)
    return tuple(found)


@lru_cache(maxsize=1 << 20)
def fours_count(s):
    """Number of fours made through the center stone (an open four counts once)."""
    found = completions(s, True, MID)
    count, k = 0, 0
    while k < len(found):
        if k + 1 < len(found) and found[k + 1] - found[k] == 5 and all(
                s[x] == 1 for x in range(found[k] + 1, found[k + 1])):
            k += 2
        else:
            k += 1
        count += 1
    return count


def _open_four(s, include):
    found = completions(s, True, include)
    return any(b - a == 5 and all(s[x] == 1 for x in range(a + 1, b)) for a, b in zip(found, found[1:]))


@lru_cache(maxsize=1 << 20)
def open_three_cells(s):
    """Indexes where a further stone makes an open four containing the center stone (empty if already a four)."""
    if completions(s, True, MID):
        return ()
    out = []
    for c, value in enumerate(s):
        if value == 0:
            t = s[:c] + b"\x01" + s[c + 1:]
            if _open_four(t, MID):
                out.append(c)
    return tuple(out)


# ---------------------------------------------------------------- black forbidden moves
def is_forbidden(code, cell, depth=0):
    """True if a black stone at the empty `cell` is forbidden. `code` is black's view and is restored."""
    code[cell] = 1
    try:
        strings = [code[LINE[d, cell]].tobytes() for d in range(4)]
        runs = [bounds(s, MID) for s in strings]
        lengths = [b - a + 1 for a, b in runs]
        if 5 in lengths:
            return False
        if max(lengths) >= 6:
            return True
        if sum(fours_count(s) for s in strings) >= 2:
            return True
        if depth >= MAX_DEPTH:
            return False
        threes = 0
        for d, s in enumerate(strings):
            for c in open_three_cells(s):
                if not is_forbidden(code, int(LINE[d, cell, c]), depth + 1):
                    threes += 1
                    break
            if threes >= 2:
                return True
        return False
    finally:
        code[cell] = 0


def forbidden_candidates(board):
    """Cheap superset of the empty cells where a black stone could possibly be forbidden."""
    black = np.append(board == 1, False).astype(np.int8)
    near = black[LINE[:, :, 2:11]].sum(-1)
    possible = ((near >= 2).sum(0) >= 2) | (near >= 4).any(0)
    return possible & (board == 0)


def forbidden_mask(board):
    """Boolean mask of the cells that are forbidden for black (independent of whose turn it is)."""
    mask = np.zeros(CELLS, dtype=bool)
    if (board == 1).sum() < 4:
        return mask
    cells = np.flatnonzero(forbidden_candidates(board))
    if len(cells):
        code = to_code(board, 1)
        for cell in cells:
            mask[cell] = is_forbidden(code, int(cell))
    return mask


def legal_mask(board, color):
    legal = board == 0
    return legal & ~forbidden_mask(board) if color == 1 else legal


# ---------------------------------------------------------------- wins and threats
def winning_cells(board, color):
    """Cells where `color` wins immediately (black: exactly five; a forbidden shape does not matter)."""
    values = board[WINDOWS]
    hit = np.flatnonzero(((values == color).sum(1) == 4) & ((values == 0).sum(1) == 1))
    mask = np.zeros(CELLS, dtype=bool)
    if not len(hit):
        return mask
    cells = WINDOWS[hit, (values[hit] == 0).argmax(1)]
    if color == -1:
        mask[cells] = True
        return mask
    code = to_code(board, 1)
    for cell, w in zip(cells.tolist(), hit.tolist()):
        d = int(WINDOW_DIRS[w])
        code[cell] = 1
        a, b = bounds(code[LINE[d, cell]].tobytes(), MID)
        code[cell] = 0
        if b - a + 1 == 5:
            mask[cell] = True
    return mask


def creates(board, color):
    """For every legal move a: the set of winning cells it would create ({a: {b, ...}}), exact for black."""
    values = board[WINDOWS]
    hit = np.flatnonzero(((values == color).sum(1) == 3) & ((values == -color).sum(1) == 0))
    graph = {}
    if not len(hit):
        return graph
    code = to_code(board, color) if color == 1 else None
    forbidden = {}
    for w in hit.tolist():
        cells = WINDOWS[w]
        empty = [int(c) for c in cells[board[cells] == 0]]
        if len(empty) != 2:
            continue
        for a, b in (empty, empty[::-1]):
            if color == 1:
                if a not in forbidden:
                    forbidden[a] = is_forbidden(code, a)
                if forbidden[a]:
                    continue
                d = int(WINDOW_DIRS[w])
                code[a] = code[b] = 1
                lo, hi = bounds(code[LINE[d, b]].tobytes(), MID)
                code[a] = code[b] = 0
                if hi - lo + 1 != 5:
                    continue
            graph.setdefault(a, set()).add(b)
    return graph


def forks(graph):
    return [a for a, bs in graph.items() if len(bs) >= 2]


def move_result(board, cell, color):
    """Outcome of the stone already on `cell`: 'win', 'forbidden' (black loses) or None."""
    code = to_code(board, color)
    lengths = [b - a + 1 for a, b in (bounds(code[LINE[d, cell]].tobytes(), MID) for d in range(4))]
    if color == -1:
        return "win" if max(lengths) >= 5 else None
    if 5 in lengths:
        return "win"
    code[cell] = 0
    return "forbidden" if is_forbidden(code, cell) else None
