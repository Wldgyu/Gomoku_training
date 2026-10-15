import unittest

import numpy as np

from . import rules
from .env import BatchBoard, LearnerEnv, force_black_center, opponent_actions
from .rules import CELLS, CENTER, SIZE


class RenjuEnv(unittest.TestCase):
    def play(self, board, moves, start_color=1):
        for i, (row, col) in enumerate(moves):
            color = start_color if i % 2 == 0 else -start_color
            board.place(np.array([row * SIZE + col]), color)

    def test_exact_five_and_overline(self):
        board = BatchBoard(1)
        self.play(board, [(7, 3), (0, 0), (7, 4), (0, 2), (7, 5), (0, 4), (7, 6), (0, 6), (7, 7)])
        self.assertEqual((board.finished[0], board.winner[0]), (True, 1))
        board = BatchBoard(1)
        self.play(board, [(7, 3), (0, 0), (7, 4), (0, 2), (7, 5), (0, 4), (7, 7), (0, 6), (7, 8), (0, 8), (7, 6)])
        self.assertEqual((board.finished[0], board.winner[0], board.forbidden_loss[0]), (True, -1, True))

    def test_white_wins_with_six(self):
        board = BatchBoard(1)
        self.play(board, [(0, 0), (2, 2), (0, 2), (2, 3), (0, 4), (2, 4), (0, 6), (2, 6), (0, 8), (2, 7), (14, 14), (2, 5)])
        self.assertEqual((board.finished[0], board.winner[0]), (True, -1))

    def test_legal_mask_hides_forbidden_cells_from_black_only(self):
        board = BatchBoard(1)
        board.board[0, [7 * SIZE + 5, 7 * SIZE + 6, 5 * SIZE + 7, 6 * SIZE + 7]] = 1
        self.assertFalse(board.legal(1)[0, CENTER])
        self.assertTrue(board.legal(-1)[0, CENTER])

    def test_center_opening_and_rule_players_never_play_forbidden(self):
        env = LearnerEnv(16, seed=3)
        rng = np.random.default_rng(0)
        for _ in range(150):
            obs, legal = env.observe()
            action = np.where(legal, rng.random(legal.shape), -1).argmax(1)
            _, done, info = env.step(action)
            self.assertFalse(env.games.forbidden_loss.any())
        board = BatchBoard(4)
        actions = opponent_actions(board, 1, np.random.default_rng(1))
        self.assertTrue((actions == CENTER).all())

    def test_draw_on_full_board(self):
        board = BatchBoard(1)
        pattern = np.array([1, 1, -1, -1] * 60)[:CELLS]
        grid = np.zeros(CELLS, dtype=np.int8)
        order = np.arange(CELLS)
        # fill all but the last cell with a drawn-looking pattern, then place the last stone
        grid[:] = np.resize([1, -1, -1, 1], CELLS)
        board.board[0] = grid
        board.board[0, -1] = 0
        board.moves[0] = CELLS - 1
        board.place(np.array([CELLS - 1]), 1)
        self.assertTrue(board.finished[0])


if __name__ == "__main__":
    unittest.main()
