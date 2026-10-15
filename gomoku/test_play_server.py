import unittest
import torch

from .play_server import Game, Application


class PlayTests(unittest.TestCase):
    def test_invalid_opening_and_occupied_move_leave_board_unchanged(self):
        game=Game('gru',1)
        with self.assertRaises(ValueError): game.place(0)
        self.assertEqual(game.moves,[])
        game.place(65)
        with self.assertRaises(ValueError): game.place(65)
        self.assertEqual(game.moves,[65])

    def test_undo_after_human_wins_restores_humans_turn(self):
        game=Game('gru',1)
        for action in [65,0,66,12,67,24,68,36,69]: game.place(action)
        self.assertTrue(game.board.finished[0])
        self.assertEqual(int(game.board.winner[0]),1)
        game.undo()
        self.assertFalse(game.board.finished[0])
        self.assertEqual(game.turn,game.human)
        self.assertEqual(len(game.moves),8)

    def test_candidate_entries_keep_originals_and_point_at_candidate_files(self):
        from .opening_followup import PATHS
        from .play_server import CHECKPOINTS, NAMES, WIDE_DIR
        for kind,path in PATHS.items(): self.assertEqual(CHECKPOINTS[kind],path)
        for kind in CHECKPOINTS:
            if kind.endswith('_quiet'):
                base=kind[:-len('_quiet')]
                self.assertEqual(CHECKPOINTS[kind],WIDE_DIR/f'scratch_{base}_w64x6_atk_q.pt')
                self.assertIn('초중반 공격 자료',NAMES[kind])
            elif kind.endswith('_wide'):
                base=kind[:-len('_wide')]
                self.assertEqual(CHECKPOINTS[kind],WIDE_DIR/f'scratch_{base}_w64x6_atk.pt')
        if (WIDE_DIR/'scratch_gru_w64x6_atk_q.pt').exists(): self.assertIn('gru_quiet',CHECKPOINTS)

    def test_real_models_play_legally_and_undo_to_human_turn(self):
        torch.set_num_threads(2)
        app=Application()
        from .play_server import CHECKPOINTS
        kinds=('gru','fly','rnn','rewired')+tuple(k for k in CHECKPOINTS if k.endswith(('_wide','_quiet')))
        for kind in kinds:
            state=app.process('/api/new',{'kind':kind,'human':-1})
            self.assertEqual(state['moves'],[65])
            state=app.process('/api/move',{'id':state['id'],'action':66})
            self.assertEqual(len(state['moves']),3)
            self.assertEqual(len(set(state['moves'])),3)
            self.assertEqual(state['turn'],-1)
            state=app.process('/api/undo',{'id':state['id']})
            self.assertEqual(state['moves'],[65])
            self.assertEqual(state['turn'],-1)


if __name__=='__main__': unittest.main()
