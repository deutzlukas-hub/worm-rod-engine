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

    def test_euler_angle_from_body_frame(self):

        # Test 1: Simple rotations
        angles = [
            [0, 0, 0],  # No rotation
            [np.pi / 2, 0, 0],  # 90° around Z
            [0, np.pi / 2, 0],  # 90° around Y
            [0, 0, np.pi / 2],  # 90° around X
            [np.pi / 4, np.pi / 4, np.pi / 4]  # 45° around each axis
        ]

        N = 100  # Number of points

        for test_angles in angles:
            # Create frame with repeated test angles
            theta = np.tile(np.array(test_angles).reshape(-1, 1), (1, N))  # Shape (3, N)

            F = Frame(theta=theta)
            # Convert angles to body frame vectors
            F.body_frame_euler_angles()

            # Store original d vectors
            d1 = F.d1.copy()
            d2 = F.d2.copy()
            d3 = F.d3.copy()

            # Check if vectors are normalized (length = 1)
            self.assertTrue(np.allclose(np.linalg.norm(d1, axis=0), 1.0))
            self.assertTrue(np.allclose(np.linalg.norm(d2, axis=0), 1.0))
            self.assertTrue(np.allclose(np.linalg.norm(d2, axis=0), 1.0))

            # Check if vectors are orthogonal (dot product = 0)
            self.assertTrue(np.allclose(np.sum(d1 * d2, axis=0), 0.0))
            self.assertTrue(np.allclose(np.sum(d2 * d3, axis=0), 0.0))
            self.assertTrue(np.allclose(np.sum(d3 * d1, axis=0), 0.0))

            # Check if right-handed (cross product d1 × d2 = d3)
            self.assertTrue(np.allclose(np.cross(d1, d2, axis=0), d3))

            # Convert back to angles
            F.euler_angles_from_body_frame()

            # Check if we got back the same angles
            self.assertTrue(np.allclose(F.theta, theta))

            # Convert these new angles back to body frame vectors
            F.body_frame_euler_angles()

            # Check if we got back the same vectors
            self.assertTrue(np.allclose(F.d1, d1, atol=1e-10))
            self.assertTrue(np.allclose(F.d2, d2, atol=1e-10))
            self.assertTrue(np.allclose(F.d3, d3, atol=1e-10))

    def test_euler_angle_from_body_frame_2(self):

        # Test 2: Random angles
        for _ in range(3):  # Run 3 random tests
            N = np.random.randint(10, 100)  # Random number of points

            alpha = np.random.uniform(-np.pi, np.pi, N)
            beta = np.random.uniform(-np.pi / 2, np.pi / 2, N)
            gamma = np.random.uniform(-np.pi /2 , np.pi / 2, N)

            # Generate random angles between -pi and pi
            theta = np.vstack((alpha, beta, gamma))

            F = Frame(theta=theta)

            # Convert to body frame vectors
            F.body_frame_euler_angles()

            # Store original vectors
            d1 = F.d1.copy()
            d2 = F.d2.copy()
            d3 = F.d3.copy()

            # Check if vectors are normalized (length = 1)
            self.assertTrue(np.allclose(np.linalg.norm(d1, axis=0), 1.0))
            self.assertTrue(np.allclose(np.linalg.norm(d2, axis=0), 1.0))
            self.assertTrue(np.allclose(np.linalg.norm(d2, axis=0), 1.0))

            # Check if vectors are orthogonal (dot product = 0)
            self.assertTrue(np.allclose(np.sum(d1 * d2, axis=0), 0.0))
            self.assertTrue(np.allclose(np.sum(d2 * d3, axis=0), 0.0))
            self.assertTrue(np.allclose(np.sum(d3 * d1, axis=0), 0.0))

            # Check if right-handed (cross product d1 × d2 = d3)
            self.assertTrue(np.allclose(np.cross(d1, d2, axis=0), d3))

            # Convert back to angles
            F.euler_angles_from_body_frame()

            # Check if we got back the same angles
            self.assertTrue(np.allclose(F.theta, theta))

            # Convert these new angles back to body frame vectors
            F.body_frame_euler_angles()

            # Verify we get back the same vectors
            self.assertTrue(np.allclose(F.d1, d1, atol=1e-10))
            self.assertTrue(np.allclose(F.d2, d2, atol=1e-10))
            self.assertTrue(np.allclose(F.d3, d3, atol=1e-10))


if __name__ == '__main__':
    unittest.main()
