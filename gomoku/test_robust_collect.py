import unittest
import numpy as np
from .env import BatchBoard
from .robust_long_plan import RobustTeacher
from .robust_collect import records_from_tree

class RobustCollectionTests(unittest.TestCase):
    def history_and_board(self):
        history=np.full(144,-1,dtype=np.int16)
        history[:6]=[61,0,62,12,63,24]
        board=BatchBoard(1,12)
        for i,a in enumerate(history[:6]):
            board.place(np.array([a]),1 if i%2==0 else -1)
        return history,board

    def test_exact_leaf_preserves_history_and_legal_targets(self):
        history,board=self.history_and_board()
        plan=RobustTeacher().solve(board.board[0],1)
        rows=records_from_tree(history,1,plan)
        self.assertEqual(len(rows),1)
        np.testing.assert_array_equal(rows[0]['histories'],history)
        np.testing.assert_array_equal(rows[0]['obs'],board.observe(1)[0])
        self.assertTrue(rows[0]['labels'].any())
        self.assertTrue(board.legal()[0,rows[0]['labels']].all())

    def test_unknown_tree_never_produces_training_records(self):
        history,_=self.history_and_board()
        with self.assertRaises(ValueError):
            records_from_tree(history,1,dict(status='unknown'))

if __name__=='__main__':
    unittest.main()
