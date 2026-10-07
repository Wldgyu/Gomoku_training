"""부분관측 단서와 에피소드 경계의 순환 상태를 확인하는 작은 회귀 검사."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from minigrid_memory_pilot import (
    GRAPH_DIR,
    MemoryPolicy,
    create_env,
    masked_cue_observation,
    matched_hidden_size,
    parameter_count,
    reset_episode,
    tensor_observation,
)


class MiniGridMemoryPilotChecks(unittest.TestCase):
    def test_fixed_start_exposes_cue_and_mask_hides_only_observation(self) -> None:
        env = create_env(11, 100)
        try:
            for seed in range(10):
                observation = reset_episode(env, seed, "fixed_cue")
                base = env.unwrapped
                cue = base.grid.get(1, base.height // 2 - 1)
                masked = masked_cue_observation(env)
                self.assertFalse(np.array_equal(observation["image"], masked["image"]))
                self.assertIs(base.grid.get(1, base.height // 2 - 1), cue)
                self.assertEqual(observation["mission"], masked["mission"])
        finally:
            env.close()

    def test_parameter_matching_and_episode_reset(self) -> None:
        torch.manual_seed(42)
        reference = MemoryPolicy("fly", 1000, GRAPH_DIR / "malecns_cx_1000.npz")
        target = parameter_count(reference)
        env = create_env(11, 100)
        try:
            observation = reset_episode(env, 42, "fixed_cue")
            image, direction = tensor_observation([observation, observation], torch.device("cpu"))
        finally:
            env.close()
        for kind in ("rnn", "gru"):
            model = MemoryPolicy(kind, 1000, hidden_size=matched_hidden_size(kind, target))
            self.assertLess(abs(parameter_count(model) - target) / target, 0.01)
            model.eval()
            images = image.unsqueeze(0).repeat(3, 1, 1, 1, 1)
            directions = direction.unsqueeze(0).repeat(3, 1)
            resets = torch.zeros(3, 2, dtype=torch.bool)
            resets[2, 0] = True
            start = model.initial_state(2, torch.device("cpu"))
            with torch.no_grad():
                logits, values = model.trajectory(images, directions, resets, start)
                state = start
                for step in range(3):
                    state = state * (~resets[step]).unsqueeze(1)
                    direct_logits, direct_values, state = model.step(
                        images[step], directions[step], state
                    )
                    torch.testing.assert_close(logits[step], direct_logits)
                    torch.testing.assert_close(values[step], direct_values)

    def test_rewiring_can_share_initial_parameters(self) -> None:
        torch.manual_seed(301)
        original_state = torch.random.get_rng_state()
        real = MemoryPolicy("fly", 1000, GRAPH_DIR / "malecns_cx_1000.npz")
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(original_state)
            rewired = MemoryPolicy("fly", 1000, GRAPH_DIR / "malecns_cx_1000_rewired_seed0.npz")
        for (real_name, real_parameter), (rewired_name, rewired_parameter) in zip(
                real.named_parameters(), rewired.named_parameters(), strict=True):
            self.assertEqual(real_name, rewired_name)
            torch.testing.assert_close(real_parameter, rewired_parameter)
        self.assertFalse(torch.equal(real.post, rewired.post))


if __name__ == "__main__":
    unittest.main()
