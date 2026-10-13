import unittest
from unittest.mock import patch,MagicMock
import numpy as np
from .coverage_teacher import RootCoverageTeacher

class CoverageTeacherTests(unittest.TestCase):
    def common(self):
        return [patch('gomoku.coverage_teacher.attack',return_value=(0,np.zeros(144,dtype=bool))),
                patch('gomoku.coverage_teacher.has_win',return_value=False),
                patch('gomoku.coverage_teacher.winning_cells',return_value=np.zeros(144,dtype=bool))]

    def test_all_replies_required(self):
        managers=self.common()
        for manager in managers:manager.start()
        try:
            with patch('gomoku.coverage_teacher.nested_quiet_candidates',return_value=[10]), \
                 patch('gomoku.coverage_teacher.opponent_replies',return_value=[11,13]), \
                 patch('gomoku.coverage_teacher.RobustTeacher') as cls:
                cls.return_value.nodes=2
                cls.return_value.solve.side_effect=[dict(status='win',kind='exact_tactic',own_moves=1,labels=[15]),dict(status='no_plan')]
                self.assertEqual(RootCoverageTeacher().solve(np.zeros(144,dtype=np.int8),1)['status'],'no_plan')
        finally:
            for manager in managers:manager.stop()

    def test_each_root_gets_fresh_budget(self):
        managers=self.common()
        for manager in managers:manager.start()
        try:
            with patch('gomoku.coverage_teacher.nested_quiet_candidates',return_value=[10,12]), \
                 patch('gomoku.coverage_teacher.opponent_replies',return_value=[11]), \
                 patch('gomoku.coverage_teacher.RobustTeacher') as cls:
                solver=MagicMock(nodes=0)
                starts=[]
                def solve(cells,color,depth):
                    starts.append(solver.nodes);solver.nodes=2000
                    if cells[10]==1:return dict(status='no_plan')
                    return dict(status='win',kind='exact_tactic',own_moves=1,labels=[15])
                solver.solve.side_effect=solve;cls.return_value=solver
                teacher=RootCoverageTeacher()
                plan=teacher.solve(np.zeros(144,dtype=np.int8),1)
                self.assertEqual(plan['labels'],[12]);self.assertEqual(teacher.nodes,4001)
                self.assertEqual(starts,[0,0]);self.assertEqual(cls.call_count,1)
        finally:
            for manager in managers:manager.stop()

if __name__=='__main__':unittest.main()
