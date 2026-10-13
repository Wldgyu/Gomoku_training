import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .history_aux import HistoryTacticalData

class RootSamplerTests(unittest.TestCase):
    def test_large_branch_does_not_dominate_small_branch(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bank.npz'
            obs=np.zeros((100,4,12,12),dtype=np.int8)
            labels=np.zeros((100,144),dtype=bool);labels[:,0]=True
            np.savez_compressed(p,obs=obs,labels=labels,category=np.ones(100,dtype=np.int8),
                                histories=np.full((100,144),-1,dtype=np.int16),keys=np.arange(100),
                                root_keys=np.array(['large']*99+['small']))
            data=HistoryTacticalData(p,torch.device('cpu'))
            torch.manual_seed(438)
            ids=data.sample_ids(10000).numpy()
            self.assertTrue(((ids>=0)&(ids<100)).all())
            self.assertTrue(4500<int((ids==99).sum())<5500)

    def test_malformed_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bank.npz'
            np.savez_compressed(p,obs=np.zeros((2,4,12,12),dtype=np.int8),labels=np.ones((2,144),dtype=bool),
                                category=np.ones(2),histories=np.full((2,144),-1),keys=np.arange(2),root_keys=np.array(['only_one']))
            with self.assertRaises(ValueError):HistoryTacticalData(p,torch.device('cpu'))

if __name__=='__main__':unittest.main()
