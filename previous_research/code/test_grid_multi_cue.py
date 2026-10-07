"""방·복도 환경의 정답, 질문 노출 시점, 지연 구성 검증."""

from __future__ import annotations

import unittest

import numpy as np

from grid_multi_cue import (
    COLOR_NAMES,
    GridSpec,
    GridMemory,
    build_dataset,
    make_layout,
    run_episode,
)

DOOR = 4  # MiniGrid object id of a door


class GridChecks(unittest.TestCase):
    def test_answer_matches_kind_of_queried_cue(self) -> None:
        for cues in (2, 3, 4):
            spec = GridSpec(cues, 8, 3)
            for seed in range(30):
                layout = make_layout(spec, np.random.default_rng(seed))
                by_color = {color: kind for _, color, kind in layout["cues"]}
                self.assertEqual(len(by_color), cues)
                self.assertEqual(by_color[layout["query_color"]], layout["answer"])
                self.assertEqual(len(layout["distractors"]), 3)

    def test_query_door_appears_only_after_the_last_cue_left_view(self) -> None:
        for cues in (2, 4):
            spec = GridSpec(cues, 6, 2)
            for seed in range(20):
                episode = run_episode(spec, seed)
                last_cue_x = episode["layout"]["cues"][-1][0]
                door_seen = [(frame[..., 0] == DOOR).any() for frame in episode["images"]]
                first_door_x = 1 + door_seen.index(True)  # one forward move per frame
                self.assertGreater(first_door_x, last_cue_x)

    def test_door_color_encodes_the_query(self) -> None:
        episode = run_episode(GridSpec(3, 8, 2), 5)
        final = episode["images"][-1]
        door_cells = final[final[..., 0] == DOOR]
        color_id = int(door_cells[0, 1])
        # minigrid color indices: red 0, green 1, blue 2, purple 3, yellow 4, grey 5
        self.assertEqual({0: "red", 1: "green", 2: "blue", 4: "yellow"}[color_id],
                         COLOR_NAMES[episode["layout"]["query_color"]])

    def test_waiting_adds_repeated_frames_without_changing_the_answer(self) -> None:
        plain = run_episode(GridSpec(2, 8, 2, 0.0), 9)
        waited = run_episode(GridSpec(2, 8, 2, 0.5), 9)
        self.assertEqual(plain["layout"]["answer"], waited["layout"]["answer"])
        self.assertGreater(len(waited["images"]), len(plain["images"]))
        self.assertTrue(np.array_equal(plain["images"][-1], waited["images"][-1]))

    def test_dataset_is_reproducible_and_balanced(self) -> None:
        specs = [GridSpec(2, 6, 2), GridSpec(2, 10, 2)]
        first = build_dataset(specs, 300, 1)
        again = build_dataset(specs, 300, 1)
        self.assertTrue((first["images"] == again["images"]).all())
        counts = first["answers"].bincount(minlength=3).float() / 300
        self.assertLess(float((counts - 1 / 3).abs().max()), 0.1)
        self.assertEqual(set(first["delay"].tolist()), {6, 10})

    def test_invalid_specs_and_forward_shape(self) -> None:
        with self.assertRaises(ValueError):
            GridSpec(2, 5, 0)
        data = build_dataset([GridSpec(2, 6, 1)], 4, 2)
        logits = GridMemory("gru", 16)(data["images"], data["lengths"])
        self.assertEqual(tuple(logits.shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
