"""Checks for observation-only attributes and checkpoint-state equivalence."""

import unittest

import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from multi_cue_memory import MemoryModel
from multi_cue_observation_recovery import (
    observation_summary,
    perception_metrics,
    visible_attributes,
)
from multi_cue_recovery import final_states


class RecoveryChecks(unittest.TestCase):
    def test_variable_length_states_match_original_output_for_all_models(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        inputs = torch.randn(3, 7, 64)
        lengths = torch.tensor([2, 5, 7])
        for kind in ("fly", "rewired", "rnn", "gru", "leaky"):
            memory = MemoryModel(kind, 16, input_dim=64)
            state = final_states(memory, inputs, torch.device("cpu"), lengths)
            torch.testing.assert_close(
                memory.head(state), memory(inputs, lengths), atol=3e-6, rtol=3e-5
            )

    def test_visible_attributes_handle_yellow_and_ignore_unseen_colors(self):
        image = torch.zeros(2, 7, 7, 3, dtype=torch.uint8)
        image[0, 1, 1] = torch.tensor([OBJECT_TO_IDX["key"], COLOR_TO_IDX["yellow"], 0])
        image[0, 2, 1] = torch.tensor(
            [OBJECT_TO_IDX["door"], COLOR_TO_IDX["yellow"], 2]
        )
        # An unseen cell's unused color field is not a visible object.
        image[1, :, :, 1] = COLOR_TO_IDX["yellow"]
        counts, query = visible_attributes(image)
        self.assertEqual(float(counts[0, 9]), 1)
        self.assertEqual(float(query[0, 3]), 1)
        self.assertEqual(float(counts[1].sum() + query[1].sum()), 0)
        features = observation_summary(image)
        self.assertEqual(features.shape, (2, 64))
        self.assertAlmostEqual(float(features[0, 9]), 1 / 3, places=6)

    def test_summary_depends_only_on_supplied_observation(self):
        image = torch.zeros(1, 7, 7, 3, dtype=torch.uint8)
        first = observation_summary(image)
        image[0, 0, 0, 0] = OBJECT_TO_IDX["wall"]
        second = observation_summary(image)
        self.assertEqual(int(first.ne(second).sum()), 1)
        image[0, 3, 6, 0] = OBJECT_TO_IDX["wall"]
        with self.assertRaises(ValueError):
            observation_summary(image)

    def test_perception_metrics_count_positive_errors(self):
        truth = torch.zeros(2, 16, dtype=torch.bool)
        truth[0, 0] = True
        truth[1, 15] = True
        predicted = truth.clone()
        predicted[0, 1] = True
        result = perception_metrics(predicted, truth)
        self.assertAlmostEqual(result["positive_f1"], 0.8)
        self.assertEqual(result["positive_recall"], 1)
        self.assertEqual(result["exact_frame_accuracy"], 0.5)
        self.assertEqual(result["visible_query_color_accuracy"], 1)
        # Forced color classification differs from detecting a positive door bit.
        absent = torch.zeros_like(truth)
        scores = torch.tensor([[-4.0, -3.0, -2.0, -1.0], [-4.0, -3.0, -2.0, -1.0]])
        result = perception_metrics(absent, truth, scores)
        self.assertEqual(result["visible_query_color_accuracy"], 1)
        self.assertEqual(result["visible_query_exact_bits"], 0)


if __name__ == "__main__":
    unittest.main()
