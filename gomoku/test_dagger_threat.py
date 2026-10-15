import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .dagger import collect as collect_dagger
from .env import BatchBoard
from .history_aux import HistoryTacticalData
from .midgame_data import collect, save
from .models import matched_hidden
from .spatial import SpatialPolicy
from .midgame_data import canonical
from .threat_aux import ThreatData, generate


class DaggerThreatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        root = Path(__file__).resolve().parent / "runs" / "midgame_20261007"
        cls.tmp = tempfile.TemporaryDirectory(dir=root)
        cls.root = Path(cls.tmp.name)
        cls.base_records = collect(2, 310_000_481)
        cls.extra_records = collect(2, 311_000_481)
        cls.base = cls.root / "base.npz"
        cls.extra = cls.root / "extra.npz"
        save(cls.base, cls.base_records)
        save(cls.extra, cls.extra_records)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_append_is_deduplicated_and_keeps_fixed_base_boundary(self):
        data = HistoryTacticalData(self.base, torch.device("cpu"), "history")
        original = len(data.obs)
        data.append_file(self.base)
        self.assertEqual(len(data.obs), original)
        data.append_file(self.extra)
        self.assertGreater(len(data.obs), original)
        after = len(data.obs)
        data.append_file(self.extra)
        self.assertEqual(len(data.obs), after)
        self.assertEqual(data.base_count, original)

    def test_candidate_conditioned_threat_loss_reaches_head_and_encoder(self):
        path = self.root / "threat.npz"
        stats = generate(path, [self.base, self.extra], 312_000_481, 16)
        self.assertGreater(stats["examples"], 0)
        model = SpatialPolicy("gru", 4911, matched_hidden("gru", 12), 12,
                              threat_aux=True)
        data = ThreatData(path, torch.device("cpu"))
        loss = data.loss(model, min(8, stats["examples"]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.threat_head[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.encoder.trunk[0].weight.grad.abs().sum()), 0)

    def test_collector_records_reconstruct_to_saved_observation(self):
        model = SpatialPolicy("gru", 4912, matched_hidden("gru", 12), 12)
        records, stats = collect_dagger(model, 8, 313_000_481, "cpu")
        self.assertEqual(stats["games"], 8)
        self.assertEqual(stats["selected_loss_records"], len(records))
        for record in records:
            board = BatchBoard(1, 12)
            moves = record["histories"]
            for ply, action in enumerate(moves[moves >= 0]):
                board.place(np.array([action]), 1 if ply % 2 == 0 else -1)
            color = 1 if int(board.moves[0]) % 2 == 0 else -1
            np.testing.assert_array_equal(board.observe(color)[0], record["obs"])
            self.assertTrue(record["labels"].any())
            self.assertGreater(record["category"], 0)

    def test_novel_collector_never_returns_reserved_failure_boards(self):
        model = SpatialPolicy("gru", 4912, matched_hidden("gru", 12), 12)
        first, _ = collect_dagger(model, 8, 313_000_481, "cpu")
        def record_key(record):
            obs = record["obs"]
            color = 1 if obs[3, 0, 0] else -1
            return canonical(((obs[0] - obs[1]) * color).reshape(-1).astype(np.int8), color)
        reserved = {record_key(r) for r in first}
        records, _ = collect_dagger(model, 8, 313_000_481, "cpu", excluded=reserved,
                                    skip_excluded_departures=True)
        self.assertFalse(reserved & {record_key(r) for r in records})


if __name__ == "__main__":
    unittest.main()

