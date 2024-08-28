# From built-in
import unittest
# From third-party
import numpy as np
from dolfin import FunctionAssigner
from fenics import Expression, Function
# From worm-rod-engine
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.worm import Worm
from worm_rod_engine.util import v2f, f2n

class TestUtil(unittest.TestCase):

    def test_v2f_f2n_round_trip_1(self):

        for _ in range(10):

            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            worm = Worm(numerical_param=numerical_param)

            arr_in = np.random.random((3, N))

            if np.random.choice([True, False]):
                func = v2f(arr_in, fs=worm.V3)
            else:
                func = Function(worm.V3)
                v2f(arr_in, func)

            arr_out = f2n(func)
            err = np.sum(np.linalg.norm(arr_in - arr_out), axis = 0)
            self.assertAlmostEqual(err, 0, msg="v2f and f2n round trip failed")

    def test_v2f_f2n_round_trip_2(self):

        for _ in range(10):

            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            worm = Worm(numerical_param=numerical_param)

            arr1_in, arr2_in = np.random.random((3, N)), np.random.random((3, N))
            x1, x2 = Function(worm.V3), Function(worm.V3)
            v2f(arr1_in, x1)
            v2f(arr2_in, x2)

            u = Function(worm.W)
            fa = FunctionAssigner(worm.W, [worm.V3, worm.V3])
            fa.assign(u, [x1, x2])
            x1_split, x2_split = u.split(deepcopy=True)
            arr1_out, arr2_out = f2n(x1_split), f2n(x2_split)

            err1 = np.sum(np.linalg.norm(arr1_in - arr1_out), axis=0)
            err2 = np.sum(np.linalg.norm(arr2_in - arr2_out), axis=0)
            self.assertAlmostEqual(err1, 0, msg="v2f and f2n round trip for x1 failed")
            self.assertAlmostEqual(err2, 0, msg="v2f and f2n round trip for x2 failed")

    def test_v2f_expr(self):

        for _ in range(10):
            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            worm = Worm(numerical_param=numerical_param)

            expr = Expression(('cos(x[0])', 'sin(x[0])', 'x[0]'), degree=1)
            func = v2f(expr, fs=worm.V3)
            arr_out = f2n(func)

            s = np.linspace(0, 1, worm.N)
            arr_expected = np.array([np.cos(s), np.sin(s), s])
            err = np.sum(np.linalg.norm(np.abs(arr_out - arr_expected)))

            self.assertAlmostEqual(err, 0)


if __name__ == '__main__':
    unittest.main()
