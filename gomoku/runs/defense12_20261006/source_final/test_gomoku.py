import unittest

import numpy as np
import torch

from .env import BatchBoard, LearnerEnv, geometry, opponent_actions
from .models import Policy, count_parameters, masked_distribution, matched_hidden, parameter_formula
from .train import advantages


class RulesTests(unittest.TestCase):
    def test_four_directions(self):
        for cells in ([0, 1, 2, 3, 4], [0, 9, 18, 27, 36],
                      [0, 10, 20, 30, 40], [8, 16, 24, 32, 40]):
            board = BatchBoard(1)
            for cell in cells[:4]:
                board.place(np.array([cell]), 1)
                self.assertFalse(board.finished[0])
            board.place(np.array([cells[4]]), 1)
            self.assertEqual(board.winner[0], 1)

    def test_overline_wins_for_both_colors(self):
        for color in (1, -1):
            board = BatchBoard(1)
            for cell in (0, 1, 2, 4, 5):
                board.place(np.array([cell]), color)
            self.assertFalse(board.finished[0])
            board.place(np.array([3]), color)
            self.assertEqual(board.winner[0], color)

    def test_double_three_and_double_four_are_legal(self):
        for surrounding in ([39, 41, 31, 49], [38, 39, 41, 22, 31, 49]):
            board = BatchBoard(1)
            for cell in surrounding + [40]:
                board.place(np.array([cell]), 1)
            self.assertFalse(board.finished[0])
            self.assertEqual(board.board[0, 40], 1)

    def test_separated_stones_and_row_wrap_do_not_win(self):
        board = BatchBoard(1)
        for cell in (6, 7, 8, 9, 10):
            board.place(np.array([cell]), 1)
        self.assertFalse(board.finished[0])

    def test_invalid_occupied_and_finished_moves_raise(self):
        board = BatchBoard(1)
        board.place(np.array([0]), 1)
        for action in (0, -1, 81):
            with self.assertRaises(ValueError):
                board.place(np.array([action]), -1)
        for cell in (1, 2, 3, 4):
            board.place(np.array([cell]), 1)
        with self.assertRaises(ValueError):
            board.place(np.array([5]), -1, np.array([True]))

    def test_full_board_draw(self):
        board = BatchBoard(1)
        for row in range(9):
            for col in range(9):
                color = 1 if (row + 2 * col) % 4 < 2 else -1
                board.place(np.array([row * 9 + col]), color)
        self.assertTrue(board.finished[0])
        self.assertEqual(board.winner[0], 0)

    def test_observation_and_reset_are_independent(self):
        board = BatchBoard(2)
        board.place(np.array([0, 1]), np.array([1, -1]))
        obs = board.observe(np.array([1, -1]))
        self.assertEqual(obs[0, 0, 0, 0], 1)
        self.assertEqual(obs[1, 0, 0, 1], 1)
        self.assertEqual(obs[1, 3].sum(), 0)
        board.reset(np.array([0]))
        self.assertEqual(board.board[0].sum(), 0)
        self.assertEqual(board.board[1, 1], -1)

    def test_tactical_win_has_priority_over_block(self):
        board = BatchBoard(1)
        for cell in (0, 1, 2, 3):
            board.place(np.array([cell]), 1)
        for cell in (9, 10, 11, 12):
            board.place(np.array([cell]), -1)
        action = opponent_actions(board, 1, np.random.default_rng(3), "tactical")
        self.assertEqual(action[0], 4)

    def test_terminal_reward_and_color_swap(self):
        env = LearnerEnv(1, seed=3)
        env.games.reset()
        for cell in (0, 1, 2, 3):
            env.games.place(np.array([cell]), 1)
        reward, done, info = env.step(np.array([4]))
        self.assertEqual(reward[0], 1)
        self.assertTrue(done[0])
        self.assertEqual(info["colors"][0], 1)
        self.assertEqual(env.colors[0], -1)
        self.assertEqual(env.games.moves[0], 1)

    def test_batch_matches_naive_rule_during_random_games(self):
        board = BatchBoard(16)
        rng = np.random.default_rng(99)
        lines, _ = geometry(9)
        for turn in range(81):
            active = ~board.finished
            if not active.any():
                break
            color = 1 if turn % 2 == 0 else -1
            board.place(opponent_actions(board, color, rng), color, active)
            for row in np.flatnonzero(active):
                naive = np.any(np.all(board.board[row, lines] == color, axis=1))
                self.assertEqual(board.winner[row] == color, naive)


class LearningTests(unittest.TestCase):
    def test_masked_policy_cannot_play_occupied(self):
        legal = torch.zeros(2, 81, dtype=torch.bool)
        legal[:, 5] = True
        dist = masked_distribution(torch.randn(2, 81), legal)
        self.assertTrue(torch.equal(dist.sample(), torch.tensor([5, 5])))

    def test_gae_stops_at_terminal(self):
        rewards = torch.tensor([[1.], [-1.]])
        values = torch.tensor([[0.2], [0.3]])
        dones = torch.tensor([[True], [True]])
        adv, target = advantages(rewards, values, dones, torch.tensor([999.]))
        self.assertTrue(torch.allclose(target, rewards))
        self.assertTrue(torch.allclose(adv, rewards - values))

    def test_matching_and_common_initialization(self):
        fly = Policy("fly", 4101)
        rewired = Policy("rewired", 4101, rewire_seed=1)
        self.assertEqual(count_parameters(fly), count_parameters(rewired))
        self.assertTrue(torch.equal(fly.edge_values, rewired.edge_values))
        self.assertTrue(torch.equal(fly.input_layer.weight, rewired.input_layer.weight))
        self.assertTrue(torch.equal(fly.actor.weight, rewired.actor.weight))
        for kind in ("fly", "rewired", "rnn", "gru"):
            h = matched_hidden(kind)
            model = Policy(kind, 4101, h)
            self.assertEqual(count_parameters(model), parameter_formula(kind, h))
            self.assertLess(abs(count_parameters(model) - count_parameters(fly)) / count_parameters(fly), 0.01)
            for x, y in zip(fly.encoder.parameters(), model.encoder.parameters()):
                self.assertTrue(torch.equal(x, y))

    def test_sequence_step_equivalence_and_state_reset(self):
        torch.set_num_threads(2)
        obs = torch.randn(3, 2, 4, 9, 9)
        resets = torch.tensor([[True, True], [False, True], [True, False]])
        for kind in ("fly", "rnn", "gru"):
            model = Policy(kind, 4101, matched_hidden(kind))
            state = torch.randn(2, model.hidden)
            p, v = model.sequence(obs, state.clone(), resets)
            for t in range(3):
                p1, v1, state = model.step(obs[t], state, resets[t])
                self.assertTrue(torch.allclose(p[t], p1, atol=1e-6))
                self.assertTrue(torch.allclose(v[t], v1, atol=1e-6))
            if kind == "fly":
                (p.sum() + v.sum()).backward()
                self.assertIsNotNone(model.edge_values.grad)
                self.assertTrue(torch.isfinite(model.edge_values.grad).all())


if __name__ == "__main__":
    unittest.main()
