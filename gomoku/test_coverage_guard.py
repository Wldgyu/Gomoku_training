import unittest
from unittest.mock import patch,MagicMock
import numpy as np
from .coverage_teacher import coverage_game

class CoverageGameGuardTests(unittest.TestCase):
    def test_proven_correct_move_does_not_end_game_analysis(self):
        cells=np.zeros(144,dtype=np.int8);cells[[0,1,2,3]]=1
        states=[(0,1,10,cells.tolist(),'certified_miss',[10]),
                (1,1,12,cells.tolist(),'unknown',[])]
        solver=MagicMock(nodes=1)
        solver.solve.return_value=dict(status='win',kind='sampled_or',labels=[20])
        with patch('gomoku.coverage_teacher.RootCoverageTeacher',return_value=solver):
            result=coverage_game(('arm',1,states))
        self.assertNotIn('selected_existing_index',result)
        self.assertEqual([r['index'] for r in result['rows']],[1])

if __name__=='__main__':unittest.main()
