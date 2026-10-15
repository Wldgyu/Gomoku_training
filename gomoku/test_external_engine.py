import sys
import unittest
import numpy as np

from .external_engine import PiskvorkEngine


def fake_command(move):
    code = "\n".join([
        "import sys",
        "for line in sys.stdin:",
        " line=line.strip()",
        " if line.startswith(('START','RESTART')): print('OK',flush=True)",
        " elif line=='END': break",
        f" elif line=='DONE': print('{move}',flush=True)",
        " elif line and line[0].isdigit(): print('MESSAGE stone '+line,flush=True)",
    ])
    return [sys.executable,"-u","-c",code]


class ExternalEngineTests(unittest.TestCase):
    def test_xy_coordinates_and_relative_color_mapping(self):
        cells=np.zeros(144,dtype=np.int8); cells[0]=1
        with PiskvorkEngine(fake_command('1,2')) as engine:
            reply=engine.recommend(cells,-1)
            self.assertEqual(reply['action'],25)
            self.assertIn('MESSAGE stone 0,0,2',reply['messages'])

    def test_occupied_engine_move_is_rejected(self):
        cells=np.zeros(144,dtype=np.int8); cells[0]=1
        with PiskvorkEngine(fake_command('0,0')) as engine:
            with self.assertRaisesRegex(RuntimeError,'occupied'):
                engine.recommend(cells,-1)

    def test_board_input_ends_with_opponent_and_matches_actual_history(self):
        cells=np.zeros(144,dtype=np.int8)
        cells[[0,10]]=1; cells[143]=-1
        with PiskvorkEngine(fake_command('1,2')) as engine:
            reply=engine.recommend(cells,-1,history=[0,143,10])
            stones=[s for s in reply['messages'] if s.startswith('MESSAGE stone')]
            self.assertEqual(stones,['MESSAGE stone 0,0,2','MESSAGE stone 11,11,1','MESSAGE stone 10,0,2'])
            with self.assertRaisesRegex(ValueError,'side to move'):
                engine.recommend(cells,1)


if __name__=='__main__':
    unittest.main()
