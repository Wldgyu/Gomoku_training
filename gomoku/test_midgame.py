"""Regression checks for curriculum resets, exact-depth wins and PPO prefixes."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .midgame_data import collect,save,attack
from .midgame_env import MidgameEnv,prefix_sequence,restore_prefix_state
from .spatial import SpatialPolicy
from .models import matched_hidden
from .history_aux import HistoryTacticalData
from .env import BatchBoard,LearnerEnv

class MidgameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent/'runs'/'midgame_20261007')
        cls.path=Path(cls.tmp.name)/'starts.npz'
        cls.records=collect(8,300_000_481)
        save(cls.path,cls.records)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_history_and_incremental_counts_match_after_resets(self):
        env=MidgameEnv(8,4,12,'tactical',self.path,0)
        for depth in (1,2,3):
            env.level=depth
            env.reset(np.arange(8),flip=False)
            for row in range(8):
                reconstructed=BatchBoard(1,12)
                moves=env.histories[row]
                for t,a in enumerate(moves[moves>=0]):
                    reconstructed.place(np.array([a]),1 if t%2==0 else -1)
                np.testing.assert_array_equal(reconstructed.board[0],env.games.board[row])
                np.testing.assert_array_equal(reconstructed.line_counts[0],env.games.line_counts[row])
                self.assertEqual(attack(env.games.board[row],int(env.colors[row]))[0],depth)
                self.assertEqual(int(env.games.moves[row])%2,int(env.colors[row]==-1))

    def test_certified_teacher_wins_within_each_declared_horizon(self):
        env=MidgameEnv(8,4,12,'tactical',self.path,0)
        for depth in (1,2,3):
            env.level=depth
            env.reset(np.arange(8),flip=False)
            finished=set()
            for _ in range(depth):
                actions=np.array([np.flatnonzero(attack(c,int(color))[1])[0]
                                  for c,color in zip(env.games.board,env.colors)])
                _,_,info=env.step(actions)
                for row,result,n in zip(info['rows'],info['results'],info['episode_decisions']):
                    if int(row) not in finished:
                        self.assertEqual(result,1)
                        self.assertLessEqual(n,depth)
                        finished.add(int(row))
            self.assertEqual(len(finished),8)

    def test_all_empty_mixture_matches_ordinary_initial_board(self):
        env=MidgameEnv(8,4,12,'tactical',self.path,1)
        standard=LearnerEnv(8,4,12,'tactical')
        np.testing.assert_array_equal(env.observe()[0],standard.observe()[0])
        self.assertFalse(env.start_horizon.any())

    def test_prefix_sequence_matches_collection_and_receives_gradient(self):
        torch.manual_seed(4)
        obs=torch.rand(3,4,4,12,12)
        histories=torch.from_numpy(np.array([r['histories'] for r in self.records[:4]]).astype(np.int64))
        prefix=histories[None].expand(3,-1,-1)
        resets=torch.tensor([[True,True,True,True],[False,True,False,True],[False,False,False,False]])
        for kind in ('fly','gru'):
            model=SpatialPolicy(kind,4810,matched_hidden(kind,12),12)
            initial=model.initial(4,torch.device('cpu'))
            logits,values=prefix_sequence(model,obs,initial,resets,prefix)
            state=initial
            ps,vs=[],[]
            for t in range(3):
                state=restore_prefix_state(model,state,resets[t],prefix[t])
                p,v,state=model.step(obs[t],state,torch.zeros(4,dtype=torch.bool))
                ps.append(p)
                vs.append(v)
            torch.testing.assert_close(logits,torch.stack(ps),atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(values,torch.stack(vs),atol=1e-6,rtol=1e-5)
            (logits.square().mean()+values.square().mean()).backward()
            recurrent=model.edge_values if kind=='fly' else model.core.weight_hh
            self.assertGreater(float(recurrent.grad.abs().sum()),0)

    def test_balanced_aux_sampling_ignores_unequal_dataset_frequencies(self):
        records=[r for r in self.records if r['horizon']==1]*8+[r for r in self.records if r['horizon']>1]
        path=Path(self.tmp.name)/'balanced_examples.npz'
        save(path,records)
        data=HistoryTacticalData(path,torch.device('cpu'),'cold')
        data.balanced=True
        model=SpatialPolicy('gru',4810,matched_hidden('gru',12),12)
        torch.manual_seed(4)
        loss=data.loss(model,192)
        self.assertTrue(torch.isfinite(loss))
        counts=list(data.samples_by_category.values())
        self.assertTrue(all(40<n<88 for n in counts),counts)
        self.assertEqual(sum(counts),192)

if __name__=='__main__':
    unittest.main()
