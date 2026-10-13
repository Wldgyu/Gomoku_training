import unittest
from unittest.mock import patch
import numpy as np
from .robust_long_plan import RobustTeacher, opponent_replies


class RobustTeacherTests(unittest.TestCase):
    def test_replies_legal_deterministic_include_win_and_block(self):
        cells = np.zeros(144, dtype=np.int8)
        cells[12:16] = -1
        cells[60:64] = 1
        replies = opponent_replies(cells, 1)
        self.assertEqual(replies, opponent_replies(cells, 1))
        self.assertEqual(len(replies), len(set(replies)))
        self.assertTrue(all(cells[r] == 0 for r in replies))
        self.assertIn(16, replies)
        self.assertIn(64, replies)

    def test_direct_fork_is_exact_leaf(self):
        cells = np.zeros(144, dtype=np.int8)
        cells[61:64] = 1
        teacher = RobustTeacher()
        plan = teacher.solve(cells, 1)
        self.assertEqual(plan['status'], 'win')
        self.assertEqual(plan['kind'], 'exact_tactic')
        self.assertEqual(plan['own_moves'], 2)

    def run_tree(self, failure=False, budget=20):
        def fake_attack(cells, color):
            labels = np.zeros(144, dtype=bool)
            if cells[2] == -1 and not failure or cells[3] == -1:
                labels[4] = True
                return 1, labels
            return 0, labels
        with patch('gomoku.robust_long_plan.attack', side_effect=fake_attack), \
             patch('gomoku.robust_long_plan.has_win', return_value=False), \
             patch('gomoku.robust_long_plan.winning_cells', return_value=np.zeros(144, dtype=bool)), \
             patch('gomoku.robust_long_plan.quiet_candidates', return_value=np.array([1])), \
             patch('gomoku.robust_long_plan.opponent_replies', return_value=[2, 3]):
            return RobustTeacher(node_budget=budget).solve(np.zeros(144, dtype=np.int8), 1, 1)

    def test_every_reply_must_succeed(self):
        self.assertEqual(self.run_tree()['status'], 'win')
        self.assertEqual(self.run_tree(failure=True)['status'], 'no_plan')

    def test_budget_exhaustion_is_not_a_positive_label(self):
        result = self.run_tree(budget=2)
        self.assertEqual(result['status'], 'unknown')
        self.assertNotIn('labels', result)

    def test_witness_preserves_both_reply_branches(self):
        result = self.run_tree()
        self.assertEqual(result['labels'], [1])
        self.assertEqual([b['reply'] for b in result['actions']['1']['branches']], [2, 3])

    def test_invalid_limits(self):
        with self.assertRaises(ValueError):
            RobustTeacher(top_k=0)


if __name__ == '__main__':
    unittest.main()
