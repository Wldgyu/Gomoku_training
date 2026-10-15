import unittest

import numpy as np

from . import rules
from .rules import SIZE, CELLS


def board_of(black=(), white=()):
    board = np.zeros(CELLS, dtype=np.int8)
    for r, c in black:
        board[r * SIZE + c] = 1
    for r, c in white:
        board[r * SIZE + c] = -1
    return board


def cell(r, c):
    return r * SIZE + c


def d4(board, s):
    grid = board.reshape(SIZE, SIZE)
    if s >= 4:
        grid = grid[:, ::-1]
    return np.ascontiguousarray(np.rot90(grid, s % 4)).reshape(-1)


class RenjuRules(unittest.TestCase):
    def forbidden(self, board, r, c):
        return bool(rules.forbidden_mask(board)[cell(r, c)])

    def test_overline_forbidden_but_exact_five_wins(self):
        board = board_of(black=[(7, 3), (7, 4), (7, 5), (7, 7), (7, 8)])
        self.assertTrue(self.forbidden(board, 7, 6))
        five = board_of(black=[(7, 3), (7, 4), (7, 5), (7, 6)])
        self.assertFalse(self.forbidden(five, 7, 7))
        after = five.copy()
        after[cell(7, 7)] = 1
        self.assertEqual(rules.move_result(after, cell(7, 7), 1), "win")
        six = board_of(black=[(7, 3), (7, 4), (7, 5), (7, 7), (7, 8)])
        six[cell(7, 6)] = 1
        self.assertEqual(rules.move_result(six, cell(7, 6), 1), "forbidden")

    def test_white_six_wins(self):
        board = board_of(white=[(2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 5)])
        self.assertEqual(rules.move_result(board, cell(2, 5), -1), "win")

    def test_double_three(self):
        board = board_of(black=[(7, 5), (7, 6), (5, 7), (6, 7)])
        self.assertTrue(self.forbidden(board, 7, 7))

    def test_dead_three_is_not_double_three(self):
        board = board_of(black=[(7, 5), (7, 6), (5, 7), (6, 7)], white=[(4, 7), (8, 7)])
        self.assertFalse(self.forbidden(board, 7, 7))
        half = board_of(black=[(7, 5), (7, 6), (5, 7), (6, 7)], white=[(8, 7), (7, 4)])
        self.assertFalse(self.forbidden(half, 7, 7))

    def test_split_three_counts(self):
        # column x x . [x] is an open split three, so together with the row three the move is a 3-3
        board = board_of(black=[(7, 5), (7, 6), (5, 7), (4, 7)])
        self.assertTrue(self.forbidden(board, 7, 7))
        # column x . x . [x] cannot become an open four, only a closed four
        board = board_of(black=[(7, 5), (7, 6), (5, 7), (3, 7)])
        self.assertFalse(self.forbidden(board, 7, 7))

    def test_double_four(self):
        board = board_of(black=[(7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7)])
        self.assertTrue(self.forbidden(board, 7, 7))

    def test_four_three_allowed(self):
        board = board_of(black=[(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)])
        self.assertFalse(self.forbidden(board, 7, 7))

    def test_double_four_in_one_line(self):
        board = board_of(black=[(7, 3), (7, 5), (7, 7), (7, 9)])
        self.assertTrue(self.forbidden(board, 7, 6))

    def test_open_four_counts_once(self):
        board = board_of(black=[(7, 4), (7, 5), (7, 6), (4, 6), (4, 9), (5, 6)])
        self.assertFalse(self.forbidden(board, 7, 7))

    def test_three_made_by_forbidden_cell_is_not_a_three(self):
        # With a white stone at (3,7) the only open-four cell of the column three is (8,7).
        base = dict(black=[(5, 7), (6, 7), (7, 5), (7, 6)], white=[(3, 7)])
        self.assertTrue(self.forbidden(board_of(**base), 7, 7))
        # Once (8,7) is itself an overline for black, the column is no longer a three.
        board = board_of(black=base["black"] + [(8, 4), (8, 5), (8, 6), (8, 8), (8, 9)], white=base["white"])
        self.assertTrue(rules.forbidden_mask(board)[cell(8, 7)])
        self.assertFalse(self.forbidden(board, 7, 7))

    def test_winning_cells_exactness(self):
        board = board_of(black=[(7, 3), (7, 4), (7, 5), (7, 6)])
        wins = np.flatnonzero(rules.winning_cells(board, 1)).tolist()
        self.assertEqual(wins, [cell(7, 2), cell(7, 7)])
        board = board_of(black=[(7, 3), (7, 4), (7, 5), (7, 6), (7, 8)])
        self.assertEqual(np.flatnonzero(rules.winning_cells(board, 1)).tolist(), [cell(7, 2)])
        white = board_of(white=[(7, 3), (7, 4), (7, 5), (7, 6), (7, 8)])
        self.assertEqual(np.flatnonzero(rules.winning_cells(white, -1)).tolist(), [cell(7, 2), cell(7, 7)])

    def test_creates_open_three(self):
        board = board_of(black=[(7, 5), (7, 6), (7, 7)])
        graph = rules.creates(board, 1)
        self.assertEqual(set(rules.forks(graph)), {cell(7, 4), cell(7, 8)})
        self.assertEqual(graph[cell(7, 4)], {cell(7, 3), cell(7, 8)})

    def test_black_fork_that_is_forbidden_is_not_offered(self):
        board = board_of(black=[(7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7)])
        graph = rules.creates(board, 1)
        self.assertNotIn(cell(7, 7), graph)

    def test_prefilter_is_sound_and_rules_are_symmetric(self):
        rng = np.random.default_rng(3)
        for trial in range(60):
            board = np.zeros(CELLS, dtype=np.int8)
            count = int(rng.integers(8, 40))
            spots = rng.choice(CELLS, count, replace=False)
            board[spots[: count // 2 + 1]] = 1
            board[spots[count // 2 + 1:]] = -1
            code = rules.to_code(board, 1)
            full = np.array([board[i] == 0 and rules.is_forbidden(code, i) for i in range(CELLS)])
            self.assertTrue(np.array_equal(full, rules.forbidden_mask(board)))
            for s in (1, 3, 5, 6):
                self.assertTrue(np.array_equal(rules.forbidden_mask(d4(board, s)), d4(full, s)))
                self.assertTrue(np.array_equal(rules.winning_cells(d4(board, s), 1),
                                               d4(rules.winning_cells(board, 1), s)))
                graph = rules.creates(board, 1)
                other = rules.creates(d4(board, s), 1)
                self.assertEqual(sorted(len(v) for v in graph.values()), sorted(len(v) for v in other.values()))


if __name__ == "__main__":
    unittest.main()
