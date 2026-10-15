import tempfile
import unittest
from pathlib import Path

import torch

from .models import label_set_nll, load_policy, matched_hidden
from .spatial import SpatialPolicy


class LabelSetLossTests(unittest.TestCase):
    def test_a_higher_wrong_logit_raises_the_loss_and_gets_a_gradient(self):
        legal = torch.tensor([[True, True]])
        labels = torch.tensor([[True, False]])
        low = label_set_nll(torch.tensor([[0., 0.]]), legal, labels)
        tied = torch.tensor([[0., 0.]], requires_grad=True)
        high = torch.tensor([[0., 5.]], requires_grad=True)
        loss = label_set_nll(high, legal, labels)
        self.assertGreater(float(loss.detach()), float(low))
        self.assertGreater(float(loss.detach()), float(label_set_nll(tied, legal, labels).detach()))
        loss.backward()
        self.assertLess(float(high.grad[0, 0]), 0)
        self.assertGreater(float(high.grad[0, 1]), 0)

    def test_missing_legal_label_is_rejected(self):
        with self.assertRaises(ValueError):
            label_set_nll(torch.zeros(1, 2), torch.tensor([[True, False]]), torch.tensor([[False, False]]))

    def test_recurrent_cell_starts_silent_then_receives_the_label_gradient(self):
        hidden = matched_hidden("gru", 12)
        plain = SpatialPolicy("gru", 4501, hidden, 12)
        headed = SpatialPolicy("gru", 4501, hidden, 12, recurrent_cell=True)
        obs = torch.rand(2, 4, 12, 12)
        state = torch.randn(2, headed.hidden)
        resets = torch.zeros(2, dtype=torch.bool)
        with torch.no_grad():
            base, _, _ = plain.step(obs, state, resets)
            current, _, _ = headed.step(obs, state, resets)
        torch.testing.assert_close(current, base)
        legal = torch.ones(2, 144, dtype=torch.bool)
        labels = torch.zeros(2, 144, dtype=torch.bool)
        labels[:, 3] = True
        logits, _, _ = headed.step(obs, state, resets)
        label_set_nll(logits, legal, labels).backward()
        self.assertGreater(float(headed.recurrent_to_cell.weight.grad.abs().sum()), 0)

    def test_recurrent_step_matches_sequence_and_reloads(self):
        hidden = matched_hidden("gru", 12)
        model = SpatialPolicy("gru", 17, hidden, 12, recurrent_cell=True)
        with torch.no_grad():
            model.recurrent_to_cell.bias.fill_(0.25)
        obs = torch.rand(3, 2, 4, 12, 12)
        resets = torch.zeros(3, 2, dtype=torch.bool)
        seq, values = model.sequence(obs, model.initial(2, torch.device("cpu")), resets)
        state = model.initial(2, torch.device("cpu"))
        steps, step_values = [], []
        for t in range(3):
            logit, value, state = model.step(obs[t], state, resets[t])
            steps.append(logit)
            step_values.append(value)
        torch.testing.assert_close(seq, torch.stack(steps))
        torch.testing.assert_close(values, torch.stack(step_values))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gru.pt"
            torch.save(dict(
                config=dict(kind="gru", seed=17, hidden=model.hidden, size=12, rewire_seed=0,
                            architecture="spatial", recurrent_cell=True),
                graph_sha256=None,
                model={key: value.detach().cpu() for key, value in model.state_dict().items()},
            ), path)
            loaded, _ = load_policy(path, torch.device("cpu"))
        self.assertTrue(loaded.use_recurrent_cell)
        torch.testing.assert_close(loaded.recurrent_to_cell.bias, model.recurrent_to_cell.bias)


if __name__ == "__main__":
    unittest.main()
