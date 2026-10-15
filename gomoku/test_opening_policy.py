import unittest

import numpy as np
import torch

from .env import BatchBoard, LearnerEnv, black_center_action, policy_action_mask
from .models import masked_distribution, Policy
from .advanced_match_audit import audit


class OpeningPolicyTests(unittest.TestCase):
    def test_forced_opening_has_executed_action_probability_one(self):
        env = LearnerEnv(2, seed=14, size=12, opponent="tactical")
        mask = torch.from_numpy(policy_action_mask(env.games, env.colors))
        logits = torch.zeros(2, 144, requires_grad=True)
        distribution = masked_distribution(logits, mask)
        action = distribution.sample()
        self.assertEqual(int(action[0]), black_center_action(12))
        self.assertAlmostEqual(float(distribution.log_prob(action)[0].detach()), 0.0)
        self.assertEqual(int(mask[0].sum()), 1)
        self.assertEqual(int(mask[1].sum()), 143)
        (-distribution.log_prob(action)[0]).backward()
        self.assertTrue(torch.equal(logits.grad[0], torch.zeros(144)))
        env.step(action.numpy())
        self.assertEqual(int(env.games.board[0, int(action[0])]), 1)
        self.assertTrue(np.array_equal(policy_action_mask(env.games, env.colors), env.games.legal()))

    def test_white_empty_board_and_nonempty_black_keep_occupancy_mask(self):
        board = BatchBoard(2, 12)
        board.place(np.array([0, 0]), 1, np.array([False, True]))
        self.assertTrue(np.array_equal(policy_action_mask(board, np.array([-1, 1])), board.legal()))

    def test_audit_replays_record_raw_and_forced_openings(self):
        torch.set_num_threads(2)
        model = Policy("rnn", 14, hidden=16, size=12)
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.copy_(-torch.arange(144, dtype=torch.float32))
        for force in (False, True):
            result, replays = audit(model, 14, games=2, record_all=True,
                                    environment_seed=286000481, force_center=force,
                                    center_opponent_opening=True)
            expected = black_center_action(12) if force else 0
            self.assertEqual(result["black_first_moves"], [expected])
            black = next(r for r in replays if r["a_color"] == 1)
            white = next(r for r in replays if r["a_color"] == -1)
            self.assertEqual(black["moves"][0], expected)
            self.assertEqual(white["moves"][0], black_center_action(12))


if __name__ == "__main__":
    unittest.main()
