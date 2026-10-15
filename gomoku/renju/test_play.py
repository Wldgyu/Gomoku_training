import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import play_server
from .rules import CENTER, SIZE
from .train import build, save_checkpoint


class RenjuPlay(unittest.TestCase):
    def test_game_flow_and_forbidden_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = build("rnn", 1, 32, 2950, channels=16, blocks=1)
            save_checkpoint(Path(tmp) / "rnn_test.pt", model, "rnn", 1, 32, 2950, 16, 1)
            play_server.MODELS = Path(tmp)
            app = play_server.Application()
            state = app.process("/api/new", dict(kind="rnn_test", human=-1))
            self.assertEqual(state["moves"], [CENTER])
            self.assertEqual(state["size"], SIZE)
            identifier = state["id"]
            move = next(i for i in range(SIZE * SIZE) if state["board"][i] == 0)
            state = app.process("/api/move", dict(id=identifier, action=move))
            self.assertEqual(len(state["moves"]), 3)
            state = app.process("/api/undo", dict(id=identifier))
            self.assertEqual(len(state["moves"]), 1)
            with self.assertRaises(ValueError):
                app.process("/api/move", dict(id=identifier, action=CENTER))
            # human black: a forbidden cell is rejected
            game = play_server.Game("rnn_test", 1)
            for row, col in ((7, 7), (0, 0), (8, 8), (0, 14), (9, 7), (14, 0), (9, 8), (14, 14)):
                game.place(row * SIZE + col)
            self.assertEqual(game.turn, 1)
            self.assertIn(9 * SIZE + 9, game.forbidden())
            with self.assertRaises(ValueError):
                game.place(9 * SIZE + 9, by_human=True)


if __name__ == "__main__":
    unittest.main()
