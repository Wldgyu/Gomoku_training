import unittest
import numpy as np
import torch

from .balanced_curriculum import STAGES, sample_batch, preservation_loss
from .models import Policy
from .advanced_match_audit import audit


class BalancedCurriculumTests(unittest.TestCase):
    def test_sampler_guarantees_defense_and_color_quotas(self):
        records = []
        for category in range(1, 7):
            for black in (True, False):
                for source in ("base", "guard", "fresh"):
                    obs = np.zeros((4, 12, 12), dtype=np.int8)
                    obs[3] = black
                    records.append(dict(obs=obs, category=category, source=source))
        for _, quotas in STAGES:
            ids, ledger = sample_batch(records, quotas, np.random.default_rng(1))
            self.assertEqual(len(ids), 128)
            self.assertEqual(sum(bool(records[i]["obs"][3, 0, 0]) for i in ids), 64)
            for group, quota in quotas.items():
                self.assertEqual(ledger[group], dict(black=quota//2, white=quota//2))
            self.assertTrue(all(records[i]["source"] == "guard" for i in ids[:quotas["guard"]]))

    def test_preservation_does_not_reinforce_wrong_defenses_or_attack(self):
        teacher = torch.tensor([[4., 0., 0.], [4., 0., 0.], [4., 0., 0.]], requires_grad=True)
        student = torch.zeros_like(teacher, requires_grad=True)
        labels = torch.tensor([[True, False, False], [False, True, False], [True, False, False]])
        loss, n = preservation_loss(student, teacher, torch.ones_like(labels), labels, torch.tensor([2, 2, 3]))
        self.assertEqual(n, 1)
        loss.backward()
        self.assertGreater(float(student.grad[0].abs().sum()), 0)
        self.assertEqual(float(student.grad[1:].abs().sum()), 0)
        self.assertIsNone(teacher.grad)

    def test_diverse_openings_preserve_replay_and_turn_order(self):
        torch.set_num_threads(2)
        model = Policy("rnn", 1, hidden=16, size=12)
        with torch.no_grad():
            model.actor.weight.zero_()
            model.actor.bias.copy_(-torch.arange(144, dtype=torch.float32))
        openings = np.array([64, 64, 66, 66])
        row, replays = audit(model, 1, games=4, record_all=True, environment_seed=291000481,
                             center_opponent_opening=True, opening_white_moves=openings)
        self.assertEqual(row["games"], 4)
        for replay in replays:
            self.assertEqual(replay["moves"][:2], [65, int(openings[replay["opening_index"]])])


if __name__ == "__main__":
    unittest.main()
