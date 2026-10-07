"""복도 선택 진단과 전체 이동 교사 경로의 누출·정답 검사."""

import unittest

import numpy as np

from minigrid_full_policy_curriculum import teacher_episode
from minigrid_memory_diagnostic import episode
from minigrid_random_start_curriculum import (
    teacher_episode as random_start_teacher_episode,
)


class MemoryStageChecks(unittest.TestCase):
    def test_mask_preserves_label_and_teacher_route_succeeds(self) -> None:
        labels = []
        for seed in range(30):
            sequence, label, cue_label = episode(seed, 11)
            masked, masked_label, masked_cue_label = episode(seed, 11, mask_cue=True)
            swapped, swapped_label, swapped_cue_label = episode(seed, 11, swap_cue=True)
            self.assertEqual((label, cue_label), (masked_label, masked_cue_label))
            self.assertEqual((label, cue_label), (swapped_label, swapped_cue_label))
            self.assertFalse(np.array_equal(sequence[0]["image"], masked[0]["image"]))
            self.assertFalse(np.array_equal(sequence[0]["image"], swapped[0]["image"]))
            self.assertEqual(len(sequence), len(masked))
            teacher_sequence, actions = teacher_episode(seed, 11)
            self.assertEqual(len(teacher_sequence), 10)
            self.assertEqual(actions, [2] * 8 + [label, 2])
            labels.append(label)
        self.assertIn(0, labels)
        self.assertIn(1, labels)

    def test_random_start_teacher_visits_cue_and_succeeds(self) -> None:
        for size in (7, 9, 11, 13):
            starts = set()
            for seed in range(100):
                observations, actions, start_x = random_start_teacher_episode(seed, size)
                self.assertEqual(len(observations), len(actions))
                self.assertEqual(actions[-1], 2)
                starts.add(start_x)
            self.assertEqual(starts, set(range(1, size - 2)))


if __name__ == "__main__":
    unittest.main()
