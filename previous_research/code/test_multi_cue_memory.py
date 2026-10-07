"""다중 단서 과제의 정답 규칙, 정보 누출, 모델 대조 설정 검증."""

from __future__ import annotations

import unittest

import torch

from multi_cue_memory import (
    COLORS,
    INPUT_DIM,
    KINDS,
    MemoryModel,
    TaskConfig,
    make_batch,
    matched_hidden,
    parameter_count,
)

CONFIG = TaskConfig(cues_min=2, cues_max=4, delay=8, distractors=4)
ROLE = COLORS + KINDS


def decode(sample: torch.Tensor):
    """Return cues [(color, kind)], distractor count, query color from one episode."""
    cues, distractors = [], 0
    for step in sample:
        color = int(step[:COLORS].argmax()) if step[:COLORS].sum() else None
        kind = int(step[COLORS:ROLE].argmax()) if step[COLORS:ROLE].sum() else None
        if step[ROLE]:
            cues.append((color, kind))
        elif step[ROLE + 1]:
            distractors += 1
    return cues, distractors, int(sample[-1, :COLORS].argmax())


class TaskChecks(unittest.TestCase):
    def test_answer_is_the_kind_of_the_queried_color(self) -> None:
        data = make_batch(CONFIG, 500, seed=1)
        for sample, answer in zip(data["inputs"], data["answers"], strict=True):
            cues, distractors, query = decode(sample)
            colors = [color for color, _ in cues]
            self.assertEqual(len(set(colors)), len(colors))  # cue colors are distinct
            self.assertTrue(2 <= len(cues) <= 4)
            self.assertEqual(distractors, 4)
            self.assertEqual(dict(cues)[query], int(answer))
            self.assertEqual(sample.shape, (CONFIG.length, INPUT_DIM))

    def test_query_step_carries_only_color_and_role(self) -> None:
        last = make_batch(CONFIG, 200, seed=2)["inputs"][:, -1]
        self.assertTrue(last[:, COLORS:ROLE].sum() == 0)
        self.assertTrue((last[:, ROLE + 3] == 1).all())

    def test_no_shortcut_from_marginals(self) -> None:
        data = make_batch(TaskConfig(3, 3, 6, 3), 6000, seed=3)
        answers = data["answers"]
        # Marginal answer distribution is uniform, and the query color alone says nothing.
        self.assertLess(abs(float(answers.bincount(minlength=KINDS).max()) / len(answers)
                            - 1 / KINDS), 0.03)
        for color in range(COLORS):
            subset = answers[data["query_color"] == color]
            self.assertLess(float(subset.bincount(minlength=KINDS).max()) / len(subset)
                            - 1 / KINDS, 0.06)
        # The last distractor's kind is also uninformative about the answer.
        last_kind = []
        for sample in data["inputs"]:
            kinds = [int(step[COLORS:ROLE].argmax()) for step in sample if step[ROLE + 1]]
            last_kind.append(kinds[-1])
        same = (torch.tensor(last_kind) == answers).float().mean()
        self.assertLess(abs(float(same) - 1 / KINDS), 0.03)

    def test_same_seed_is_reproducible_and_seeds_differ(self) -> None:
        first = make_batch(CONFIG, 50, seed=5)["inputs"]
        self.assertTrue(torch.equal(first, make_batch(CONFIG, 50, seed=5)["inputs"]))
        self.assertFalse(torch.equal(first, make_batch(CONFIG, 50, seed=6)["inputs"]))

    def test_invalid_configs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TaskConfig(cues_min=1, cues_max=5)
        with self.assertRaises(ValueError):
            TaskConfig(delay=2, distractors=3)


class ModelChecks(unittest.TestCase):
    def test_leaky_has_no_trainable_edges_and_matches_fly_dynamics_otherwise(self) -> None:
        leaky, fly = MemoryModel("leaky", 1000), MemoryModel("fly", 1000)
        self.assertEqual(int(torch.count_nonzero(leaky.edge_values)), 0)
        self.assertFalse(leaky.edge_values.requires_grad)
        self.assertGreater(parameter_count(fly), parameter_count(leaky))
        self.assertEqual(leaky.hidden, fly.hidden)

    def test_rnn_gru_parameter_counts_match_fly(self) -> None:
        target = parameter_count(MemoryModel("fly", 1000))
        for kind in ("rnn", "gru"):
            count = parameter_count(MemoryModel(kind, matched_hidden(kind, target)))
            self.assertLess(abs(count - target) / target, 0.02)

    def test_forward_shapes_and_edge_direction(self) -> None:
        inputs = make_batch(CONFIG, 8, seed=7)["inputs"]
        for kind in ("fly", "rewired", "rnn", "gru", "leaky"):
            self.assertEqual(MemoryModel(kind, 32)(inputs).shape, (8, KINDS))
        fly = MemoryModel("fly", 1000)
        matrix = fly.matrix().detach()
        pre, post = int(fly.pre[0]), int(fly.post[0])
        self.assertNotEqual(float(matrix[post, pre]), 0.0)  # W[post, pre]


if __name__ == "__main__":
    unittest.main()
