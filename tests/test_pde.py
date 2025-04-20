# From built-in
import unittest
import warnings
# From third-party
import numpy as np
from fenics import project
# From worm-rod-engine
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.worm import Worm
from worm_rod_engine.util import v2f, f2n
from worm_rod_engine.pde.util import finite_backwards_difference, finite_difference_coefficients

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

fdc_expected = {
    (1, 1): np.array([-1.0, 1.0]),
    (1, 2): np.array([0.5, -2.0, 3.0/2.0]),
    (1, 3): np.array([-1.0/3.0, 3.0/2.0, -3.0, 11.0/6.0]),
    (1, 4): np.array([1.0/4.0, -4.0/3.0, +3.0, 4.0, 25.0/12.0])
}

class TestPDE(unittest.TestCase):

    def test_finite_difference_coefficients(self):

        for _ in range(100):

            n = 1
            k = np.random.randint(1, 4)
            c_arr, s_arr = finite_difference_coefficients(n, k)
            c_arr_expected = fdc_expected[(n, k)]
            s_arr_expected = np.arange(-(k+n-1), 1, 1)

            err_c = np.sum(np.abs(c_arr - c_arr_expected))
            err_s = np.sum(np.abs(s_arr - s_arr_expected))

            self.assertAlmostEqual(err_c, 0.0)
            self.assertAlmostEqual(err_s, 0.0)

    def test_finite_backwards_difference(self):

        for _ in range(10):

            N = np.random.randint(100, 1000)
            dt = np.random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

            numerical_param = numerical_argument_parser.parse_args(['--dt', str(dt), '--N', str(N)])
            worm = Worm(numerical_param=numerical_param)

            t_arr = np.arange(0, 10*worm.dt+0.1*worm.dt, worm.dt)
            n_t_step = len(t_arr)
            s_arr = np.linspace(0.0, 1.0, worm.N)

            u_arr = []
            u_expected_arr = np.zeros((n_t_step, worm.N))

            for i, t in enumerate(t_arr):
                arr = np.cos(s_arr + t)
                u_arr.append(v2f(arr, fs=worm.PDE.V))
                u_expected_arr[i, :] = arr

            for i in range(n_t_step):
                u = u_arr[i]
                err = np.sum(np.abs(f2n(u) - u_expected_arr[i, :]))
                self.assertAlmostEqual(err, 0.0)

                n = 1
                k = np.random.randint(1, 4)

                for i in range(k, n_t_step):

                    u_old_arr = u_arr[i-k:i]
                    u = u_arr[i]
                    u_t = finite_backwards_difference(n, k, u, u_old_arr, worm.dt)
                    u_t_arr = f2n(project(u_t, worm.PDE.V))

                    c_arr, s_arr = finite_difference_coefficients(n, k)

                    u_t_arr_expected = np.zeros_like(u_t_arr)

                    for j, idx in enumerate(s_arr + i):
                        u_t_arr_expected += c_arr[j]*u_expected_arr[idx, :]

                    u_t_arr_expected /= worm.dt

                    err = np.sum(np.abs(u_t_arr - u_t_arr_expected))

                    self.assertAlmostEqual(err, 0.0)


if __name__ == '__main__':

    unittest.main()
