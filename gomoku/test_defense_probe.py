import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import defense_probe as P
from .env import BatchBoard
from .multistep_tactics import multi_targets

DATA = P.DATA / "synthetic_test.npz"


@unittest.skipUnless(DATA.exists(), "defense data not collected")
class AugmentationTests(unittest.TestCase):
    def test_every_symmetry_keeps_history_observation_and_teacher_labels_consistent(self):
        data = P.load(DATA, augment=True)
        total = len(data["category"])
        original = total // 8
        rng = np.random.default_rng(3)
        for index in rng.choice(original, 12, replace=False):
            for sym in range(8):
                row = sym * original + index
                moves = data["histories"][row].cpu().numpy()
                moves = moves[moves >= 0]
                board = BatchBoard(1, 12)
                for ply, action in enumerate(moves):
                    board.place(np.array([action]), 1 if ply % 2 == 0 else -1)
                color = 1 if len(moves) % 2 == 0 else -1
                self.assertTrue(np.array_equal(board.observe(color)[0], data["obs"][row].cpu().numpy()))
                labels, category = multi_targets(board, np.array([color]))
                self.assertEqual(int(category[0]), int(data["category"][row]))
                self.assertTrue(np.array_equal(labels[0], data["labels"][row].cpu().numpy()))

    def test_validation_split_keeps_symmetries_together(self):
        data = P.load(DATA, augment=True)
        train, valid = P.split_validation(data)
        self.assertEqual(len(train["category"]) % 8, 0)
        self.assertEqual(len(valid["category"]) % 8, 0)
        self.assertEqual(len(train["category"]) + len(valid["category"]), len(data["category"]))


class WidePolicyTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_step(self):
        from .models import load_policy
        from .wide_spatial import WideSpatialPolicy
        model = WideSpatialPolicy("gru", 5, 32, 12, 0, 16, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wide.pt"
            torch.save(dict(config=dict(kind="gru", seed=5, hidden=32, size=12, rewire_seed=0,
                                        architecture="spatial_wide", channels=16, blocks=2),
                            graph_sha256=None, model=model.state_dict()), path)
            loaded, _ = load_policy(path, torch.device("cpu"))
        obs = torch.zeros(2, 4, 12, 12)
        state = loaded.initial(2, obs.device)
        logits, value, new_state = loaded.step(obs, state, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (2, 144))
        self.assertEqual(tuple(new_state.shape), (2, 32))
        sequence, values = loaded.sequence(obs[None].repeat(3, 1, 1, 1, 1), state, torch.zeros(3, 2, dtype=torch.bool))
        self.assertEqual(tuple(sequence.shape), (3, 2, 144))

    def test_defense_label_loss_prefers_correct_cell(self):
        logits = torch.zeros(1, 144)
        legal = torch.ones(1, 144, dtype=torch.bool)
        labels = torch.zeros(1, 144, dtype=torch.bool)
        labels[0, 7] = True
        from .models import label_set_nll
        base = label_set_nll(logits, legal, labels)
        logits[0, 7] = 5
        self.assertLess(label_set_nll(logits, legal, labels), base)


if __name__ == "__main__":
    unittest.main()
