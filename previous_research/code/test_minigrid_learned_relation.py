"""Checks for the learned MiniGrid cue memory and comparison experiment."""

import unittest

import torch

from minigrid_learned_relation import LearnedRelation, features


class FrameDetector(torch.nn.Module):
    def forward(self, images, directions):
        marker = images[:, 0, 0, 0].float() - 2
        cue = torch.stack((torch.zeros_like(marker), marker, -marker), dim=1)
        upper = torch.stack((marker, -marker), dim=1)
        return cue, upper, upper


class LearnedRelationTests(unittest.TestCase):
    def test_actor_trace_uses_recorded_branch_frame(self):
        images = torch.zeros(6, 7, 7, 3, dtype=torch.long)
        images[:, 0, 0, 0] = torch.arange(6)
        row = {"images": images, "directions": torch.zeros(6, dtype=torch.long),
               "actions": torch.zeros(6, dtype=torch.long), "branch_index": 1,
               "meta": {"correct_turn": 0, "cue_type": "Key"}, "seed": 1}
        result = features([row], FrameDetector(), torch.device("cpu"))
        self.assertEqual(result["branch"].item(), 1)
        self.assertEqual(result["cue"].shape, (1, 2, 2))
        self.assertEqual(result["upper"].argmax(dim=-1).item(), 1)

    def test_sequence_and_online_states_agree(self):
        torch.manual_seed(8)
        model = LearnedRelation("rnn", 16, 8)
        cues = torch.rand(3, 5, 2)
        branch = torch.tensor([1, 3, 4])
        upper = torch.rand(3, 2)
        turn, cue = model(cues, branch, upper)
        state = torch.zeros(3, model.hidden)
        picked = torch.zeros_like(state)
        for step in range(cues.shape[1]):
            state = model.memory_step(cues[:, step], state)
            picked = torch.where((branch == step).unsqueeze(1), state, picked)
        self.assertTrue(torch.allclose(turn, model.turn_from_state(picked, upper)))
        self.assertTrue(torch.allclose(cue, model.cue_readout(picked)))

    def test_fly_and_rewired_trainable_initialization_is_paired(self):
        torch.manual_seed(601)
        fly = LearnedRelation("fly", 1000, 601)
        torch.manual_seed(601)
        rewired = LearnedRelation("rewired", 1000, 601)
        self.assertFalse(torch.equal(fly.pre, rewired.pre)
                         and torch.equal(fly.post, rewired.post))
        for (name, parameter), (other_name, other_parameter) in zip(
            fly.named_parameters(), rewired.named_parameters()
        ):
            self.assertEqual(name, other_name)
            self.assertTrue(torch.equal(parameter, other_parameter), name)


if __name__ == "__main__":
    unittest.main()
