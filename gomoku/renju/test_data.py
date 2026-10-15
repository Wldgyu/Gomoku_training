import unittest
from pathlib import Path

import numpy as np

from . import tactics
from .rules import CELLS, SIZE

RUN = Path(__file__).resolve().parents[1] / "runs/renju15_20261015"


def check_file(case, path, sample=200):
    with np.load(path) as saved:
        obs, labels, legal, category, histories = (saved[k] for k in ("obs", "labels", "legal", "category", "histories"))
    for i in range(min(sample, len(obs))):
        color = 1 if obs[i, 3, 0, 0] else -1
        board = np.zeros(CELLS, dtype=np.int8)
        moves = [int(m) for m in histories[i] if m >= 0]
        for ply, move in enumerate(moves):
            board[move] = 1 if ply % 2 == 0 else -1
        case.assertEqual(color, 1 if len(moves) % 2 == 0 else -1)
        absolute = ((obs[i, 0] - obs[i, 1]) * color).reshape(-1)
        case.assertTrue(np.array_equal(board, absolute), (path.name, i))
        case.assertTrue(labels[i].any() or category[i] == 7)
        case.assertTrue((labels[i] & ~legal[i]).sum() == 0 or category[i] == 1)
        if category[i] in (1, 2, 3, 4, 5, 6) and i < 40:
            recomputed, cat = tactics.analyze(board, color)
            case.assertEqual(int(cat), int(category[i]))
            case.assertTrue(np.array_equal(recomputed, labels[i]))


class RenjuData(unittest.TestCase):
    def test_saved_records_are_consistent(self):
        found = [p for p in (RUN / "data").glob("*.npz")] + list((RUN / "smoke").glob("*onpolicy*.npz"))
        if not found:
            self.skipTest("no generated data yet")
        for path in found:
            check_file(self, path)


if __name__ == "__main__":
    unittest.main()
