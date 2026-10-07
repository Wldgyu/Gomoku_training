"""결합 입력·보조 목표·쓰기 게이트 개입의 정확성 검증."""

from __future__ import annotations

import unittest

import torch

from multi_cue_binding import (
    VARIANTS,
    BindMemory,
    conjunction,
    count_parameters,
    cue_targets,
    episode_descriptors,
    final_jobs,
    loss_for,
    matched_hidden,
    tune_jobs,
    without_gate,
)
from multi_cue_memory import COLORS, KINDS, MemoryModel, TaskConfig, make_batch

CONFIG = TaskConfig(3, 3, 8, 4)


class BindingChecks(unittest.TestCase):
    def test_cue_targets_match_the_episode_and_query_answer(self) -> None:
        data = make_batch(CONFIG, 200, seed=1)
        target, present = cue_targets(data["inputs"])
        self.assertTrue((present.sum(1) == 3).all())
        rows = torch.arange(200)
        self.assertTrue(present[rows, data["query_color"]].all())
        self.assertTrue(target[rows, data["query_color"]].eq(data["answers"]).all())

    def test_descriptors_find_the_queried_slot_and_reused_distractor_color(self) -> None:
        data = make_batch(CONFIG, 400, seed=2)
        facts = episode_descriptors(data["inputs"])
        self.assertTrue((facts["cues"] == 3).all())
        self.assertEqual(set(facts["slot"].tolist()), {0, 1, 2})
        inputs = data["inputs"]
        for row in range(0, 400, 40):
            cue_colors = [int(s[:COLORS].argmax()) for s in inputs[row] if s[COLORS + KINDS]]
            self.assertEqual(cue_colors[int(facts["slot"][row])], int(data["query_color"][row]))
            reused = any(int(s[:COLORS].argmax()) == int(data["query_color"][row])
                         for s in inputs[row] if s[COLORS + KINDS + 1])
            self.assertEqual(bool(facts["reused"][row]), reused)

    def test_conjunction_is_zero_without_an_object(self) -> None:
        inputs = make_batch(CONFIG, 20, seed=3)["inputs"]
        joint = conjunction(inputs)
        self.assertEqual(joint.shape[-1], COLORS * KINDS)
        self.assertEqual(float(joint[:, -1].abs().sum()), 0.0)  # query step has no kind
        self.assertTrue((joint[:, 0].sum(-1) == 1).all())  # a cue step has one pair

    def test_base_variant_matches_the_earlier_model_exactly(self) -> None:
        inputs = make_batch(CONFIG, 8, seed=4)["inputs"]
        for kind in ("fly", "leaky", "rnn", "gru"):
            torch.manual_seed(7)
            old = MemoryModel(kind, 32)
            torch.manual_seed(7)
            new = BindMemory(kind, 32, "base")
            self.assertTrue(torch.allclose(old(inputs), new(inputs)[0], atol=1e-6), kind)

    def test_initial_gate_equals_the_fixed_leak_update(self) -> None:
        inputs = make_batch(CONFIG, 8, seed=5)["inputs"]
        torch.manual_seed(9)
        plain = BindMemory("leaky", 1000, "base")
        torch.manual_seed(9)
        gated = BindMemory("leaky", 1000, "gate")
        z = torch.sigmoid(gated.gate_layer(inputs[:, 0]))
        self.assertLess(abs(float(z.mean().detach()) - 0.1), 0.03)
        self.assertEqual(plain(inputs)[0].shape, gated(inputs)[0].shape)

    def test_gate_is_rejected_for_recurrent_cells_and_removed_by_helper(self) -> None:
        with self.assertRaises(ValueError):
            BindMemory("gru", 32, "gate")
        for name in VARIANTS:
            flags = VARIANTS[without_gate(name)]
            self.assertFalse(flags[1])
            self.assertEqual((flags[0], flags[2]), (VARIANTS[name][0], VARIANTS[name][2]))

    def test_aux_loss_uses_only_shown_cue_colors_and_backpropagates(self) -> None:
        data = make_batch(CONFIG, 16, seed=6)
        model = BindMemory("gru", 32, "bind_aux")
        loss = loss_for(model, data["inputs"], data["answers"])
        loss.backward()
        self.assertGreater(float(model.aux_head.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_parameter_matching_includes_the_added_parts(self) -> None:
        for variant in ("base", "bind_aux"):
            target = count_parameters("fly", 1000, variant)
            for kind in ("rnn", "gru"):
                hidden = matched_hidden(kind, variant, target)
                self.assertLess(abs(count_parameters(kind, hidden, variant) - target) / target,
                                0.02)

    def test_job_lists_are_complete_and_unique(self) -> None:
        tune = tune_jobs()
        self.assertEqual(len({job.name for job in tune}), len(tune))
        self.assertEqual(len(tune), 88)
        final = final_jobs("gate_aux")
        self.assertEqual(len({job.name for job in final}), len(final))
        self.assertEqual(len(final), 4 * 5 * 5 * 2)
        self.assertFalse(any("gate" in job.variant for job in final if job.kind in ("rnn", "gru")))


if __name__ == "__main__":
    unittest.main()
