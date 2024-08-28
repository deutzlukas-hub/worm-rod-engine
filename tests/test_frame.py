# From built-in
import unittest
# From third-party
import numpy as np
from fenics import Expression
# From worm-rod-engine
from worm_rod_engine.parameter.output_parameter import default_output_parameter, output_parameter_types
from worm_rod_engine.frame import Frame, FrameSequence

class TestFrame(unittest.TestCase):

    def test_Frame_random(self):

        for _ in range(10):
            N = np.random.randint(100, 1000)
            frame_keys = {}

            for key in vars(default_output_parameter).keys():
                if np.random.choice([True, False]):
                    if output_parameter_types[key] == float:
                        v = np.random.rand()
                    else:
                        v = np.random.rand(3, N)
                    frame_keys[key] = v

            F = Frame(**frame_keys)
            for key in frame_keys:
                if output_parameter_types[key] == float:
                    err = np.abs(getattr(F, key) - frame_keys[key])
                else:
                    err = np.linalg.norm(getattr(F, key) - frame_keys[key], axis=0).sum()
                self.assertAlmostEqual(err, 0.0)

    def test_FrameSequence_random(self):

        for _ in range(10):
            n_t_step = np.random.randint(1, 5000)
            N = np.random.randint(100, 1000)

            frame_key_sequence = {}
            for key in vars(default_output_parameter).keys():

                if np.random.choice([True, False]):
                    if output_parameter_types[key] == float:
                        v = np.random.rand(n_t_step)
                    else:
                        v = np.random.rand(n_t_step, 3, N)
                    frame_key_sequence[key] = v

            frames = []
            for i in range(n_t_step):
                frame_key = {}
                for key, v in frame_key_sequence.items():
                    if output_parameter_types[key] == float:
                        v = frame_key_sequence[key][i]
                    else:
                        v = frame_key_sequence[key][i, :]
                    frame_key[key] = v
                frames.append(Frame(**frame_key))

            FS = FrameSequence(frames)

            for key in frame_key_sequence.keys():
                err = np.linalg.norm(getattr(FS, key) - frame_key_sequence[key], axis=0).sum()
                self.assertAlmostEqual(err, 0.0)


if __name__ == '__main__':
    unittest.main()
