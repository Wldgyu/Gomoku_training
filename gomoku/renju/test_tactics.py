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

    def deep_board(self):
        # found by the roll-out collector: black wins by three fours in a row (no 1-2 four chain, no fork)
        return board_of(black=[(2, 0), (4, 5), (4, 6), (4, 7), (5, 5), (5, 8), (6, 4), (6, 7), (7, 5), (7, 6),
                               (7, 7), (8, 6), (9, 6), (13, 6)],
                        white=[(4, 2), (4, 4), (4, 9), (4, 13), (5, 7), (6, 6), (6, 8), (7, 3), (7, 4), (7, 8),
                               (8, 5), (8, 7), (9, 7), (11, 6)])

    def test_deep_chain_needs_depth_three(self):
        board = self.deep_board()
        self.assertEqual(tactics.analyze(board, 1)[1], 0)
        self.assertEqual(tactics.deep_chains(board, 1, 2), [])
        moves = tactics.deep_chains(board, 1, 4)
        self.assertEqual(sorted(moves), [cell(2, 8), cell(3, 5), cell(3, 7)])
        labels, category = tactics.analyze_deep(board, 1)
        self.assertEqual((category, cells_of(labels)), (5, sorted(moves)))

    def test_deep_defence_leaves_no_chain(self):
        board = -self.deep_board()
        labels, category = tactics.analyze_deep(board, 1)
        self.assertEqual(category, 6)
        self.assertTrue((labels & ~rules.legal_mask(board, 1)).sum() == 0)
        for move in cells_of(labels):
            after = board.copy()
            after[move] = 1
            self.assertFalse(tactics.deep_chains(after, -1, 4))

    def test_deep_chains_cover_exact_chains(self):
        board = self.chain_board()
        self.assertTrue(set(tactics.forcing_chains(board, 1)) <= set(tactics.deep_chains(board, 1, 4)))

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

    def test_forced_win_beats_defending_against_a_three(self):
        # From a real game (RNN v5, human black): black owns an open three but white wins first with
        # a four at (6,11) [forced reply (5,12)] followed by an open four at (6,8).
        moves = [112, 128, 111, 113, 97, 127, 141, 126, 125, 83, 98, 99, 115, 129, 130, 84, 114]
        board = np.zeros(CELLS, dtype=np.int8)
        for ply, move in enumerate(moves):
            board[move] = 1 if ply % 2 == 0 else -1
        self.assertTrue(rules.forks(rules.creates(board, 1)))  # black's fork threat exists
        labels, category = tactics.analyze(board, -1)
        self.assertEqual(category, 5)
        self.assertIn(5 * SIZE + 10, cells_of(labels))  # row 6, column 11 (0-based 5, 10)

if __name__ == "__main__":
    unittest.main()
