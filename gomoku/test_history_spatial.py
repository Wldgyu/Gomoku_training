import unittest
from pathlib import Path

import numpy as np
import torch

from .env import BatchBoard
from .history_aux import burn_in, prefix_observations, HistoryTacticalData
from .models import Policy, matched_hidden, count_parameters
from .spatial import SpatialPolicy
from .expanded_tactics import expanded_targets
from .tactics import targets


class HistorySpatialTests(unittest.TestCase):
    def test_open_three_creation_and_prevention_have_real_double_wins(self):
        for color, expected in ((1, 3), (-1, 4)):
            board = BatchBoard(1, 12)
            for cell in (64, 65, 66):
                board.place(np.array([cell]), color)
            label, category = expanded_targets(board, np.array([1]))
            self.assertEqual(int(category[0]), expected)
            self.assertEqual(np.flatnonzero(label[0]).tolist(), [63, 67])
            for action in np.flatnonzero(label[0]):
                altered = BatchBoard(1, 12)
                for cell in (64, 65, 66):
                    altered.place(np.array([cell]), color)
                altered.place(np.array([action]), 1)
                if expected == 3:
                    wins, cat = targets(altered, np.array([1]))
                    self.assertEqual(cat[0], 1)
                    self.assertGreaterEqual(int(wins.sum()), 2)
                else:
                    # Brute force every opponent reply to independently verify prevention.
                    for reply in np.flatnonzero(altered.legal()[0]):
                        test = BatchBoard(1, 12)
                        for cell in (64, 65, 66):
                            test.place(np.array([cell]), -1)
                        test.place(np.array([action]), 1)
                        test.place(np.array([reply]), -1)
                        wins, cat = targets(test, np.array([-1]))
                        self.assertLess(int(wins.sum()), 2)

    def test_immediate_win_and_block_override_fork_labels(self):
        board = BatchBoard(1, 12)
        for cell in (0, 1, 2, 3):
            board.place(np.array([cell]), -1)
        for cell in (64, 65, 66):
            board.place(np.array([cell]), 1)
        label, category = expanded_targets(board, np.array([1]))
        self.assertEqual(category[0], 2)
        self.assertEqual(np.flatnonzero(label[0]).tolist(), [4])

    def test_crossing_double_four_uses_distinct_winning_cells(self):
        board = BatchBoard(1, 12)
        black = (61, 62, 63, 28, 40, 52)
        white = (60, 16, 100, 110, 120, 130)
        for b, w in zip(black, white):
            board.place(np.array([b]), 1)
            board.place(np.array([w]), -1)
        label, category = expanded_targets(board, np.array([1]))
        self.assertEqual(category[0], 3)
        self.assertEqual(np.flatnonzero(label[0]).tolist(), [64])
        board.place(np.array([64]), 1)
        winning, cat = targets(board, np.array([1]))
        self.assertEqual(cat[0], 1)
        self.assertEqual(np.flatnonzero(winning[0]).tolist(), [65, 76])

    def test_fast_history_matches_actual_player_turns(self):
        histories = np.full((4, 144), -1, dtype=np.int64)
        for row, length in enumerate((0, 1, 8, 9)):
            histories[row, :length] = np.arange(length)
        tensor = torch.from_numpy(histories)
        obs, active = prefix_observations(tensor, 12)
        for row, length in enumerate((0, 1, 8, 9)):
            board = BatchBoard(1, 12)
            color = 1 if length % 2 == 0 else -1
            decision = 0
            for ply in range(length):
                if (1 if ply % 2 == 0 else -1) == color:
                    np.testing.assert_array_equal(obs[decision, row].numpy(), board.observe(color)[0])
                    self.assertTrue(active[decision, row])
                    decision += 1
                board.place(np.array([histories[row, ply]]), 1 if ply % 2 == 0 else -1)
            self.assertEqual(decision, int(active[:, row].sum()))
        for cls in (Policy, SpatialPolicy):
            for kind in ("fly", "gru"):
                model = cls(kind, 4500, matched_hidden(kind, 12), 12)
                actual = burn_in(model, tensor)
                state = model.initial(4, torch.device("cpu"))
                with torch.no_grad():
                    for t in range(len(obs)):
                        _, _, new = model.step(obs[t], state, torch.zeros(4, dtype=torch.bool))
                        state = torch.where(active[t, :, None], new, state)
                torch.testing.assert_close(actual, state, atol=1e-6, rtol=1e-5)

    def test_spatial_sequence_and_step_are_equal_and_core_gets_gradient(self):
        torch.manual_seed(9)
        for kind in ("fly", "rewired", "rnn", "gru"):
            model = SpatialPolicy(kind, 4500, matched_hidden(kind, 12), 12)
            obs = torch.rand(3, 2, 4, 12, 12)
            reset = torch.zeros(3, 2, dtype=torch.bool)
            seq, value = model.sequence(obs, model.initial(2, torch.device("cpu")), reset)
            state = model.initial(2, torch.device("cpu"))
            ps, vs = [], []
            for t in range(3):
                p, v, state = model.step(obs[t], state, reset[t])
                ps.append(p)
                vs.append(v)
            torch.testing.assert_close(seq, torch.stack(ps), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(value, torch.stack(vs), atol=1e-6, rtol=1e-5)
            self.assertEqual(tuple(seq.shape), (3, 2, 144))
            (value.square().mean() + seq.square().mean()).backward()
            recurrent = model.edge_values if kind in ("fly", "rewired") else model.core.weight_hh
            self.assertIsNotNone(recurrent.grad)
            self.assertGreater(float(recurrent.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
