import copy,unittest
import numpy as np
from .midgame_data import attack
from .robust_long_plan import opponent_replies,verify_plan

class ProofMemoTests(unittest.TestCase):
    def tree(self):
        cells=np.zeros(144,dtype=np.int8);cells[61:64]=1;cells[[0,30,110]]=-1
        after=cells.copy();after[64]=1
        branches=[]
        for reply in opponent_replies(after,1):
            child=after.copy();child[reply]=-1
            horizon,labels=attack(child,1)
            self.assertEqual(horizon,1)
            branches.append(dict(reply=reply,plan=dict(status='win',kind='exact_tactic',own_moves=1,labels=np.flatnonzero(labels).tolist())))
        choice=dict(status='win',kind='sampled_and',own_moves=2,branches=branches)
        return cells,dict(status='win',kind='sampled_or',own_moves=2,labels=[64],actions={'64':choice})

    def test_valid_tree_and_changed_child_are_distinguished(self):
        cells,plan=self.tree()
        self.assertTrue(verify_plan(cells,1,plan))
        bad=copy.deepcopy(plan);bad['actions']['64']['branches'][0]['plan']['labels']=[61]
        self.assertFalse(verify_plan(cells,1,bad))

    def test_missing_reply_still_rejected(self):
        cells,plan=self.tree();plan['actions']['64']['branches'].pop()
        self.assertFalse(verify_plan(cells,1,plan))

if __name__=='__main__':unittest.main()
