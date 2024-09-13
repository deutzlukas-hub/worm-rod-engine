
# From built-in
import unittest
# From third-party
import numpy as np
from fenics import Expression, Function, Constant
# From worm-rod-engine
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.parameter.output_parameter import output_parameter_parser
from worm_rod_engine.worm import Worm
from worm_rod_engine.util import v2f, f2n
from worm_rod_engine.frame import Frame, FrameSequence

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*product.*")

class TestWorm(unittest.TestCase):

    def test_initialise(self):

        for _ in range(10):

            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            worm = Worm(numerical_param=numerical_param)

            r0_arr_in = np.random.rand(3, worm.N)
            theta0_arr_in = np.random.rand(3, worm.N)
            F0 = Frame(r=r0_arr_in, theta=theta0_arr_in, t=0.0)

            worm.initialise(F0=F0)

            for u in worm.PDE.u_old_arr:
                r0, theta0 = u.split(deepcopy=True)
                r0_arr_out, theta0_arr_out = f2n(r0), f2n(theta0)

                err_r = np.linalg.norm(r0_arr_in - r0_arr_out, axis=0).sum()
                err_theta = np.linalg.norm(theta0_arr_in - theta0_arr_out).sum()

                self.assertAlmostEqual(err_r, 0.0)
                self.assertAlmostEqual(err_theta, 0.0)

        return

    def test_zero_control(self):

        for dim in [2, 3]:
            numerical_param = numerical_argument_parser.parse_args(['--dt', '0.01', '--N', '250'])
            worm = Worm(dimension=dim, numerical_param=numerical_param)
            output = worm.solve(5, progress=True)
            self.assertTrue(output[0])
            FS = output[1]
            err = np.linalg.norm(FS.r[0] - FS.r[-1], axis=0).sum()
            self.assertAlmostEqual(0.0, err, places=2)


    def test_constant_curvature(self):

        for dim in [2, 3]:
            numerical_param = numerical_argument_parser.parse_args(['--dt', '0.01', '--N', '250'])
            output_param = output_parameter_parser.parse_args(['--k', '--eps'])
            worm = Worm(dimension=dim, numerical_param=numerical_param, output_param=output_param)
            k_inp = np.pi
            if dim == 2:
                k0 = Constant(k_inp)
            else:
                k0 = Constant((k_inp, 0.0, 0.0))

            output = worm.solve(2, k0=k0, progress=True)
            self.assertTrue(output[0])
            FS = output[1]

            k_out_avg = FS.k[-1, 0].mean()
            self.assertAlmostEqual(k_inp, k_out_avg, places=1)

    def test_solve_twice(self):

        for dim in [2, 3]:
            numerical_param = numerical_argument_parser.parse_args(['--dt', '0.01', '--N', '250'])
            worm = Worm(dimension=dim, numerical_param=numerical_param)
            if dim == 2:
                k0 = Expression('A*sin(q*x[0]-2*pi*t)', degree=1, A=2*np.pi, q=2*np.pi, t=0.0)
            else:
                k0 = Expression(('A*sin(q*x[0]-2*pi*t)', '0', '0'), degree=1, A=2 * np.pi, q=2 * np.pi, t=0.0)

            output1 = worm.solve(5, k0=k0, progress=True)
            self.assertTrue(output1[0])
            FS = output1[1]
            r1 = FS.r
            output2 = worm.solve(5, k0=k0, progress=True)
            self.assertTrue(output2[0])
            FS = output2[1]
            r2 = FS.r
            err = np.abs(r1 - r2).flatten().sum()
            self.assertAlmostEqual(err, 0.0)

    def test_update_against_solve(self):

        for dim in [2, 3]:
            numerical_param = numerical_argument_parser.parse_args(['--dt', '0.01', '--N', '250'])
            worm = Worm(dimension=dim, numerical_param=numerical_param)
            if dim == 2:
                k0 = Expression('A*sin(q*x[0]-2*pi*t)', degree=1, A=2*np.pi, q=2*np.pi, t=0.0)
            else:
                k0 = Expression(('A*sin(q*x[0]-2*pi*t)', '0', '0'), degree=1, A=2 * np.pi, q=2 * np.pi, t=0.0)

            output1 = worm.solve(5, k0=k0, progress=True)
            self.assertTrue(output1[0])
            FS = output1[1]
            r1 = FS.r

            worm.initialise(k0=k0)
            frames = []

            for t in FS.t:
                worm.update_state(assemble=True)
                frames.append(Frame(**worm.assembler.output))

            FS = FrameSequence(frames)
            r2 = FS.r
            err = np.abs(r1 - r2).sum()
            self.assertAlmostEqual(err, 0.0)

        return

    # def test_solve_update(self):
    #     pass
    # def test_solve(self):
    #
    #     worm = Worm()
    #
    #     A0, lam0 = 2*np.pi, 1.0
    #     q0 = 2*np.pi/lam0
    #
    #     k0 = Expression('A0*sin(q0*x[0]-2*pi*t)', degree=1, t=0.0, A0 = A0, q0=q0)
    #
    #     output = worm.solve(5, k0, progress=True, minimal_working_examples=True)
    #     self.assertTrue(output[0], msg='Solve did not finsh')
    #
    # def test_solve_expr_input_versus_arr_input(self):
    #
    #     worm = Worm()
    #
    #     A0, lam0 = 2*np.pi, 1.0
    #     q0 = 2*np.pi/lam0
    #
    #     k0 = Expression('A0*sin(q0*x[0]-2*pi*t)', degree=1, t=0.0, A0 = A0, q0=q0)
    #
    #     T_sim = 5
    #     n_t_step = int(round(T_sim/worm.dt))
    #     t_arr = np.arange()
    #     s_arr = np.linspace(0, 1, worm.N, endpoint=True)
    #
    #     k0_arr = np.zeros(n_t_step, 3, worm.N)
    #     k0_arr[:, 0, :] = A0 * np.sin(q0 * s_arr - 2*np.pi*t)
    #
    #     output = worm.solve(5, k0, progress=True)
    #     self.assertTrue(output[0], msg='Solve did not finsh')
    #
    # def test_solve_again(self):
    #
    #     worm = Worm()
    #
    #     T_sim = 5
    #     A0, lam0 = 2*np.pi, 1.0
    #     q0 = 2*np.pi/lam0
    #
    #     k0 = Expression('A0*sin(q0*x[0]-2*pi*t)', degree=1, t=0.0, A0 = A0, q0=q0)
    #
    #     output1 = worm.solve(T_sim, k0, progress=True)
    #     output2 = worm.solve(T_sim, k0, progress=True)
    #
    #     FS1, FS2 = output1[1], output2[1]
    #
    #     self.assertEqual(FS1.r, FS2.r, msg='Solve again does not yields same output')
    #
    #
    # def test_update(self):
    #
    #     pass
    # def test_constant_control(self):
    #
    #     pass
    #
    # def test_zero_control(self):
    #     pass


if __name__ == '__main__':

    unittest.main()
