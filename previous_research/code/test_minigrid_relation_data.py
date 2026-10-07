"""짝 에피소드의 정답·경로와 제외 길이 검증."""

from __future__ import annotations

import unittest

from minigrid_relation_data import (
    EXCLUDED_JUNCTION_X,
    VARIANTS,
    build_dataset,
    teacher_episode,
)


class RelationEpisodeChecks(unittest.TestCase):
    def test_factorial_interventions_flip_only_the_expected_choice(self) -> None:
        for size in (9, 15):
            for seed in (100, 101):
                episodes = [teacher_episode(seed, size, cue_flip=cue, target_flip=target)
                            for cue, target in VARIANTS]
                base_turn = episodes[0][2]["correct_turn"]
                for (cue, target), (_, actions, meta) in zip(VARIANTS, episodes, strict=True):
                    self.assertEqual(meta["correct_turn"], base_turn ^ cue ^ target)
                    self.assertEqual(actions[:-2], episodes[0][1][:-2])
                    self.assertEqual(actions[-2:], [meta["correct_turn"], 2])

    def test_variable_length_training_excludes_both_holdout_corridors(self) -> None:
        rows = build_dataset(range(100, 140), 17, variants=VARIANTS,
                             random_length=True, excluded_junction_x=EXCLUDED_JUNCTION_X)
        self.assertGreater(len(rows), 0)
        self.assertEqual(len(rows) % 4, 0)
        self.assertTrue(all(row["meta"]["junction_x"] not in EXCLUDED_JUNCTION_X for row in rows))
        self.assertTrue(all(len(row["images"]) == len(row["actions"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
