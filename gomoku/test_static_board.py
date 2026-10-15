import unittest
import torch
from .models import matched_hidden
from .spatial import SpatialPolicy
from .static_board import StaticBoardPolicy

class StaticBoardTests(unittest.TestCase):
    def test_same_board_ignores_previous_state_and_resets(self):
        base=SpatialPolicy('rnn',11,hidden=16)
        model=StaticBoardPolicy(base,4)
        obs=torch.randn(2,4,12,12)
        zero=model.initial(2,torch.device('cpu'))
        a,_,carry=model.step(obs,zero,torch.zeros(2,dtype=torch.bool))
        b,_,_=model.step(obs,torch.randn_like(zero),torch.ones(2,dtype=torch.bool))
        torch.testing.assert_close(a,b)
        self.assertEqual(int(carry.count_nonzero()),0)

    def test_one_step_matches_cold_original(self):
        base=SpatialPolicy('gru',12,hidden=16)
        model=StaticBoardPolicy(base,1)
        obs=torch.randn(2,4,12,12)
        state=base.initial(2,torch.device('cpu'))
        reset=torch.zeros(2,dtype=torch.bool)
        a,v,_=model.step(obs,state,reset)
        b,w,_=base.step(obs,state,reset)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(v,w)

    def test_fly_graph_influences_repeated_static_processing(self):
        base=SpatialPolicy('fly',13)
        obs=torch.randn(2,4,12,12)
        for repeats in (1,2):
            base.zero_grad(set_to_none=True)
            model=StaticBoardPolicy(base,repeats)
            _,value,_=model.step(obs,base.initial(2,torch.device('cpu')),torch.zeros(2,dtype=torch.bool))
            value.sum().backward()
            grad=float(base.edge_values.grad.abs().sum())
            self.assertEqual(grad==0,repeats==1)

    def test_sequence_is_independent_board_processing(self):
        base=SpatialPolicy('rnn',14,hidden=16)
        model=StaticBoardPolicy(base,3)
        obs=torch.randn(2,2,4,12,12)
        state=model.initial(2,torch.device('cpu'))
        resets=torch.zeros(2,2,dtype=torch.bool)
        logits,_=model.sequence(obs,state,resets)
        expected=torch.stack([model.step(frame,state,resets[0])[0] for frame in obs])
        torch.testing.assert_close(logits,expected)

if __name__=='__main__':
    unittest.main()
