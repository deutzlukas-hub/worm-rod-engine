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

    def test_body_frame_from_euler_angle(self):

        # Test straight configuration in z-direction
        for _ in range(3):
            N = np.random.randint(100, 1000)
            s = np.linspace(0, 1, N)
            r = np.zeros((3, N))
            r[2, :] = s
            theta = np.zeros((3, N))
            F = Frame(r=r, theta=theta)
            F.body_frame_euler_angles()

            d1_pred = np.array([1, 0, 0])
            d2_pred = np.array([0, 1, 0])
            d3_pred = np.array([0, 0, 1])

            self.assertTrue(np.allclose(F.d1 - d1_pred[:, None], np.zeros_like(F.d1)))
            self.assertTrue(np.allclose(F.d2 - d2_pred[:, None], np.zeros_like(F.d2)))
            self.assertTrue(np.allclose(F.d3 - d3_pred[:, None], np.zeros_like(F.d3)))

        # Test straight configuration in x-direction
        for _ in range(3):

            N = np.random.randint(100, 1000)
            s = np.linspace(0, 1, N)
            r = np.zeros((3, N))
            r[0, :] = s
            theta = np.zeros((3, N))
            theta[1, :] = -np.pi / 2

            F = Frame(r=r, theta=theta)
            F.body_frame_euler_angles()

            d1_pred = np.array([0, 0, -1])
            d2_pred = np.array([0, 1, 0])
            d3_pred = np.array([1, 0, 0])

            self.assertTrue(np.allclose(F.d1 - d1_pred[:, None], np.zeros_like(F.d1)))
            self.assertTrue(np.allclose(F.d2 - d2_pred[:, None], np.zeros_like(F.d2)))
            self.assertTrue(np.allclose(F.d3 - d3_pred[:, None], np.zeros_like(F.d3)))

        # Test straight configuration in y-direction
        for _ in range(3):

            N = np.random.randint(100, 1000)
            s = np.linspace(0, 1, N)
            r = np.zeros((3, N))
            r[1, :] = s
            theta = np.zeros((3, N))
            theta[2, :] = np.pi / 2

            F = Frame(r=r, theta=theta)
            F.body_frame_euler_angles()

            d1_pred = np.array([1, 0, 0])
            d2_pred = np.array([0, 0, -1])
            d3_pred = np.array([0, 1, 0])

            self.assertTrue(np.allclose(F.d1 - d1_pred[:, None], np.zeros_like(F.d1)))
            self.assertTrue(np.allclose(F.d2 - d2_pred[:, None], np.zeros_like(F.d2)))
            self.assertTrue(np.allclose(F.d3 - d3_pred[:, None], np.zeros_like(F.d3)))












if __name__ == '__main__':
    unittest.main()
