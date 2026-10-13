"""Tests for independently verified forcing chains and full-game separation."""
import unittest
import numpy as np
from .env import BatchBoard, geometry
from .multistep_tactics import forcing_chains, safe_defenses, multi_targets, winning_cells

def brute_wins(cells,color):
    lines,_=geometry(12)
    result=[]
    for a in np.flatnonzero(cells==0):
        after=cells.copy()
        after[a]=color
        if ((after[lines]==color).sum(1)==5).any():
            result.append(int(a))
    return result

class MultiStepTests(unittest.TestCase):
    def test_wins_agree_with_independent_placements(self):
        rng=np.random.default_rng(8)
        for _ in range(12):
            cells=rng.choice(np.array([-1,0,1],dtype=np.int8),144,p=[.15,.7,.15])
            for color in (1,-1):
                if ((cells[geometry(12)[0]] == color).sum(1) == 5).any():
                    continue
                self.assertEqual(np.flatnonzero(winning_cells(cells,color)).tolist(),brute_wins(cells,color))

    def test_forcing_witnesses_survive_every_reply(self):
        # A forcing line whose blocked extension leaves a crossing open three.
        cells=np.zeros(144,dtype=np.int8)
        cells[[61,62,63,40,52]]=1
        cells[[60,16,100,110,120]]=-1
        chains=forcing_chains(cells,1)
        self.assertTrue(chains)
        for a,(b,c) in chains.items():
            after=cells.copy()
            after[a]=1
            self.assertEqual(brute_wins(after,1),[b])
            self.assertFalse(brute_wins(after,-1))
            # Every reply except the forced block allows immediate victory.
            for reply in np.flatnonzero(after==0):
                branch=after.copy()
                branch[reply]=-1
                if reply!=b:
                    self.assertTrue(brute_wins(branch,1))
                    continue
                branch[c]=1
                self.assertGreaterEqual(len(brute_wins(branch,1)),2)
                self.assertFalse(brute_wins(branch,-1))
                for final_reply in np.flatnonzero(branch==0):
                    last=branch.copy()
                    last[final_reply]=-1
                    self.assertTrue(brute_wins(last,1))

    def test_defense_labels_destroy_all_chains(self):
        cells=np.zeros(144,dtype=np.int8)
        cells[[61,62,63,40,52]]=-1
        cells[[60,16,100,110,120]]=1
        self.assertTrue(forcing_chains(cells,-1))
        safe=safe_defenses(cells,1)
        self.assertTrue(safe.any())
        for a in np.flatnonzero(safe):
            after=cells.copy()
            after[a]=1
            self.assertFalse(brute_wins(after,-1))
            self.assertFalse(forcing_chains(after,-1))

    def test_holdout_reserves_both_players_turns(self):
        from .mine_multistep import game_keys
        from .tactics import position_key
        game = dict(size=12, a_color=1, winner=1,
                    moves=[0,12,1,13,2,14,3,15,4])
        keys = game_keys([game])
        board = BatchBoard(1,12)
        for ply, action in enumerate(game["moves"]):
            color = 1 if ply % 2 == 0 else -1
            self.assertIn(position_key(board.board[0],color),keys)
            board.place(np.array([action]),color)
        self.assertEqual(len(keys),9)

    def test_actual_fork_is_measured_before_opponent_reply(self):
        from .advanced_match_audit import RecordedEnv
        env = RecordedEnv(1,seed=8,size=12,opponent="tactical")
        env.games.reset()
        env.colors[:] = 1
        for b,w in zip((61,62,63,28,40,52),(60,16,100,110,120,130)):
            env.games.place(np.array([b]),1)
            env.games.place(np.array([w]),-1)
        _,_,info = env.step(np.array([64]))
        self.assertTrue(info["safe_fork"][0])

    def test_unanswerable_double_win_is_not_supervised(self):
        board=BatchBoard(1,12)
        for a in (64,65,66,67):
            board.place(np.array([a]),-1)
        labels,category=multi_targets(board,np.array([1]))
        self.assertEqual(int(category[0]),0)
        self.assertFalse(labels.any())

if __name__=='__main__':
    unittest.main()
