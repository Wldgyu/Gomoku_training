import unittest

import numpy as np

from . import rules, tactics
from .rules import CELLS, SIZE
from .test_rules import board_of, cell, d4


def cells_of(mask):
    return sorted(np.flatnonzero(mask).tolist())


class RenjuTactics(unittest.TestCase):
    def test_win(self):
        board = board_of(black=[(7, 3), (7, 4), (7, 5), (7, 6)], white=[(0, 0), (0, 2), (2, 0)])
        labels, category = tactics.analyze(board, 1)
        self.assertEqual(category, 1)
        self.assertEqual(cells_of(labels), [cell(7, 2), cell(7, 7)])

    def test_block_and_black_cannot_block_on_forbidden_cell(self):
        board = board_of(black=[(7, 3), (7, 12)], white=[(7, 8), (7, 9), (7, 10), (7, 11)])
        labels, category = tactics.analyze(board, 1)
        self.assertEqual((category, cells_of(labels)), (2, [cell(7, 7)]))
        # (7,7) is a double three for black and the only cell that stops white's four
        board = board_of(black=[(7, 5), (7, 6), (5, 7), (6, 7)],
                         white=[(4, 10), (5, 9), (6, 8), (8, 6)])
        self.assertTrue(rules.forbidden_mask(board)[cell(7, 7)])
        self.assertEqual(tactics.analyze(board, 1)[1], 0)
        # white may always occupy a cell that is forbidden for black
        self.assertEqual(tactics.analyze(board, -1)[1], 1)

    def test_create_fork(self):
        board = board_of(black=[(7, 5), (7, 6), (7, 7)], white=[(0, 0), (0, 5)])
        labels, category = tactics.analyze(board, 1)
        self.assertEqual((category, cells_of(labels)), (3, [cell(7, 4), cell(7, 8)]))

    def test_prevent_fork(self):
        board = board_of(black=[(2, 2), (11, 11), (12, 3)], white=[(7, 5), (7, 6), (7, 7)])
        labels, category = tactics.analyze(board, 1)
        self.assertEqual((category, cells_of(labels)), (4, [cell(7, 4), cell(7, 8)]))

    def chain_board(self):
        return board_of(black=[(7, 4), (7, 5), (7, 6), (5, 5), (6, 6)], white=[(7, 3)])

    def test_forcing_chain_and_its_defense(self):
        board = self.chain_board()
        labels, category = tactics.analyze(board, 1)
        self.assertEqual(category, 5)
        self.assertTrue(labels[cell(7, 7)])
        mirrored = -board
        labels, category = tactics.analyze(mirrored, 1)
        self.assertEqual(category, 6)
        self.assertTrue(labels[cell(7, 7)] or labels[cell(7, 8)])
        for move in cells_of(labels):
            after = mirrored.copy()
            after[move] = 1
            self.assertFalse(tactics.forcing_chains(after, -1))
            self.assertFalse(rules.forks(rules.creates(after, -1)))

    def test_labels_are_equivariant(self):
        rng = np.random.default_rng(5)
        for _ in range(12):
            board = np.zeros(CELLS, dtype=np.int8)
            spots = rng.choice(CELLS, 14, replace=False)
            board[spots[:7]] = 1
            board[spots[7:]] = -1
            labels, category = tactics.analyze(board, 1)
            for s in (2, 5):
                other, other_category = tactics.analyze(d4(board, s), 1)
                self.assertEqual(category, other_category)
                self.assertTrue(np.array_equal(other, d4(labels, s)))

    def test_labels_are_legal(self):
        rng = np.random.default_rng(11)
        for _ in range(40):
            board = np.zeros(CELLS, dtype=np.int8)
            spots = rng.choice(CELLS, 20, replace=False)
            board[spots[:10]] = 1
            board[spots[10:]] = -1
            for color in (1, -1):
                labels, category = tactics.analyze(board, color)
                if category not in (0, 1):
                    self.assertTrue((labels & ~rules.legal_mask(board, color)).sum() == 0)


if __name__ == "__main__":
    unittest.main()
