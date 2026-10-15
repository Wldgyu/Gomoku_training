import unittest
from unittest.mock import patch
import numpy as np
from .robust_long_plan import RobustTeacher

class WideRootTests(unittest.TestCase):
    def test_only_first_own_turn_is_widened(self):
        def candidates(cells,color,limit):
            return np.array([10 if cells[10]==0 else 12])
        def replies(cells,*args):
            return [11 if cells[11]==0 else 13]
        with patch('gomoku.robust_long_plan.attack',return_value=(0,np.zeros(144,dtype=bool))), \
             patch('gomoku.robust_long_plan.has_win',return_value=False), \
             patch('gomoku.robust_long_plan.winning_cells',return_value=np.zeros(144,dtype=bool)), \
             patch('gomoku.robust_long_plan.quiet_candidates',side_effect=candidates) as spy, \
             patch('gomoku.robust_long_plan.opponent_replies',side_effect=replies):
            RobustTeacher(root_candidate_limit=12,node_budget=20).solve(np.zeros(144,dtype=np.int8),1,2)
            self.assertEqual([call.args[2] for call in spy.call_args_list],[3,6,12,3])

    def test_default_and_explicit_three_are_identical(self):
        cells=np.zeros(144,dtype=np.int8)
        cells[[61,62]]=1;cells[[0,12]]=-1
        a=RobustTeacher(node_budget=80)
        b=RobustTeacher(node_budget=80,root_candidate_limit=3)
        self.assertEqual(a.solve(cells,1,2),b.solve(cells,1,2))
        self.assertEqual(a.nodes,b.nodes)


    def test_wider_root_retains_proof_when_budget_is_tight(self):
        def candidates(cells,color,limit):
            return np.array([10,12,14] if limit==3 else [4,6,8,10,12,14])
        def attack(cells,color):
            labels=np.zeros(144,dtype=bool)
            if cells[10]==1 and cells[11]==-1:labels[15]=True;return 1,labels
            return 0,labels
        with patch('gomoku.robust_long_plan.attack',side_effect=attack), \
             patch('gomoku.robust_long_plan.has_win',return_value=False), \
             patch('gomoku.robust_long_plan.winning_cells',return_value=np.zeros(144,dtype=bool)), \
             patch('gomoku.robust_long_plan.quiet_candidates',side_effect=candidates), \
             patch('gomoku.robust_long_plan.opponent_replies',return_value=[11]):
            for width in (3,6):
                plan=RobustTeacher(root_candidate_limit=width,node_budget=2).solve(np.zeros(144,dtype=np.int8),1,1)
                self.assertEqual(plan['status'],'win')
                self.assertIn(10,plan['labels'])

    def test_nonpositive_root_width_rejected(self):
        with self.assertRaises(ValueError):RobustTeacher(root_candidate_limit=0)

if __name__=='__main__':unittest.main()
