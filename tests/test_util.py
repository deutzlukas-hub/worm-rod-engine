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

    #================================================================================================
    # Test3D
    #================================================================================================
    def test_v2f_f2n_round_trip_1(self):

        for _ in range(10):

            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            dimension = np.random.choice([2, 3])
            worm = Worm(dimension=dimension, numerical_param=numerical_param)
            fs = worm.PDE.V2 if dimension == 2 else worm.PDE.V3
            arr_in = np.random.random((dimension, N))

            if np.random.choice([True, False]):
                func = v2f(arr_in, fs=fs)
            else:
                func = Function(fs)
                v2f(arr_in, func)

            arr_out = f2n(func)
            err = np.sum(np.linalg.norm(arr_in - arr_out), axis = 0)
            self.assertAlmostEqual(err, 0, msg="v2f and f2n round trip failed")

    def test_v2f_f2n_round_trip_2(self):

        for _ in range(10):

            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            dimension = np.random.choice([2, 3])
            worm = Worm(dimension=dimension, numerical_param=numerical_param)
            fs1 = worm.PDE.V2 if dimension == 2 else worm.PDE.V3
            fs2 = worm.PDE.V if dimension == 2 else worm.PDE.V3
            dim1 = 2 if dimension == 2 else 3
            dim2 = 1 if dimension == 2 else 3

            arr1_in, arr2_in = np.random.random((dim1, N)), np.random.random((dim2, N))
            if dim2 == 1: arr2_in = arr2_in.flatten()

            x1, x2 = Function(fs1), Function(fs2)
            v2f(arr1_in, x1)
            v2f(arr2_in, x2)

            u = Function(worm.PDE.W)
            fa = FunctionAssigner(worm.PDE.W, [fs1, fs2])
            fa.assign(u, [x1, x2])
            x1_split, x2_split = u.split(deepcopy=True)
            arr1_out, arr2_out = f2n(x1_split), f2n(x2_split)

            err1 = np.sum(np.linalg.norm(arr1_in - arr1_out), axis=0)
            err2 = np.sum(np.linalg.norm(arr2_in - arr2_out), axis=0)
            self.assertAlmostEqual(err1, 0, msg="v2f and f2n round trip for x1 failed")
            self.assertAlmostEqual(err2, 0, msg="v2f and f2n round trip for x2 failed")

    def test_v2f_expr_3D(self):

        for _ in range(10):
            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            worm = Worm(numerical_param=numerical_param)

            expr = Expression(('cos(x[0])', 'sin(x[0])', 'x[0]'), degree=1)
            func = v2f(expr, fs=worm.PDE.V3)
            arr_out = f2n(func)

            s = np.linspace(0, 1, worm.N)
            arr_expected = np.array([np.cos(s), np.sin(s), s])
            err = np.sum(np.linalg.norm(np.abs(arr_out - arr_expected)))

            self.assertAlmostEqual(err, 0)

    def test_v2f_expr_2D(self):

        for _ in range(10):
            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            worm = Worm(dimension=2, numerical_param=numerical_param)

            expr1 = Expression(('cos(x[0])', 'sin(x[0])'), degree=1)
            expr2 = Expression('cos(x[0])', degree=1)

            func = v2f(expr1, fs=worm.PDE.V2)
            arr_out_1 = f2n(func)

            func = v2f(expr2, fs=worm.PDE.V)
            arr_out_2 = f2n(func)

            s = np.linspace(0, 1, worm.N)
            arr_expected_1 = np.array([np.cos(s), np.sin(s)])
            arr_expected_2 = np.cos(s)
            err1 = np.sum(np.linalg.norm(np.abs(arr_out_1 - arr_expected_1)))
            err2 = np.sum(np.abs(arr_out_2 - arr_expected_2))
            self.assertAlmostEqual(err1, 0)
            self.assertAlmostEqual(err2, 0)

if __name__ == '__main__':
    unittest.main()
