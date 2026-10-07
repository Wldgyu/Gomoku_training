import unittest
from pathlib import Path

import numpy as np
import torch

from .env import BatchBoard, LearnerEnv
from .models import Policy, matched_hidden, count_parameters, parameter_formula
from .tactics import targets, TacticalData


class DefenseTests(unittest.TestCase):
    def test_tactical_targets_prioritize_winning_over_block(self):
        board = BatchBoard(1, 12)
        for b, w in zip((0, 1, 2, 3), (12, 13, 14, 15)):
            board.place(np.array([b]), 1)
            board.place(np.array([w]), -1)
        labels, category = targets(board, np.array([1]))
        self.assertEqual(category[0], 1)
        self.assertEqual(np.flatnonzero(labels[0]).tolist(), [4])

    def test_unique_block_and_unavoidable_double_threat(self):
        board = BatchBoard(1, 12)
        for cell in (1, 2, 3, 4):
            board.place(np.array([cell]), -1)
        _, category = targets(board, np.array([1]))
        self.assertEqual(category[0], 0)
        board.place(np.array([0]), 1)
        labels, category = targets(board, np.array([1]))
        self.assertEqual(category[0], 2)
        self.assertEqual(np.flatnonzero(labels[0]).tolist(), [5])

    def test_12x12_terminal_bound_and_shapes(self):
        env = LearnerEnv(4, seed=99, size=12, opponent="weak")
        model = Policy("gru", 99, matched_hidden("gru", 12), 12)
        obs, legal = env.observe()
        logits, _, _ = model.step(torch.from_numpy(obs), model.initial(4, torch.device("cpu")),
                                  torch.ones(4, dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (4, 144))
        env.step(logits.masked_fill(~torch.from_numpy(legal), -1e9).argmax(1).numpy())
        self.assertEqual(count_parameters(model), parameter_formula("gru", model.hidden, 12))

    def test_data_split_labels_and_reachable_histories(self):
        root = Path(__file__).resolve().parent / "runs/defense12_20261006/data"
        if not (root / "train.npz").exists():
            self.skipTest("Generated data not yet present")
        keys = []
        for name in ("train", "validation", "test"):
            with np.load(root / f"{name}.npz") as data:
                keys.append(set(data["keys"].tolist()))
                self.assertEqual(len(keys[-1]), len(data["keys"]))
                for i in np.linspace(0, len(data["category"]) - 1, 32, dtype=int):
                    board = BatchBoard(1, 12)
                    moves = data["histories"][i]
                    moves = moves[moves >= 0]
                    for ply, move in enumerate(moves):
                        self.assertFalse(board.finished[0])
                        board.place(np.array([move]), 1 if ply % 2 == 0 else -1)
                    color = 1 if len(moves) % 2 == 0 else -1
                    np.testing.assert_array_equal(board.observe(color)[0], data["obs"][i])
                    label, category = targets(board, np.array([color]))
                    np.testing.assert_array_equal(label[0], data["labels"][i])
                    self.assertEqual(category[0], data["category"][i])
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])


if __name__ == "__main__":
    unittest.main()
