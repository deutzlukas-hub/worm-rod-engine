
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
from worm_rod_engine.frame import Frame

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

            for u in worm.u_old_arr:
                r0, theta0 = u.split(deepcopy=True)
                r0_arr_out, theta0_arr_out = f2n(r0), f2n(theta0)

                err_r = np.linalg.norm(r0_arr_in - r0_arr_out, axis=0).sum()
                err_theta = np.linalg.norm(theta0_arr_in - theta0_arr_out).sum()

                self.assertAlmostEqual(err_r, 0.0)
                self.assertAlmostEqual(err_theta, 0.0)

        return

    def test_zero_control(self):

        numerical_param = numerical_argument_parser.parse_args(['--dt', '0.01', '--N', '250'])
        worm = Worm(numerical_param=numerical_param)
        output = worm.solve(5, progress=True)
        self.assertTrue(output[0])
        FS = output[1]
        err = np.linalg.norm(FS.r[0] - FS.r[-1], axis=0).sum()
        self.assertAlmostEqual(0.0, err, places=2)


    def test_constant_control_1(self):

        numerical_param = numerical_argument_parser.parse_args(['--dt', '0.01', '--N', '250'])
        output_param = output_parameter_parser.parse_args(['--k', '--eps'])
        worm = Worm(numerical_param=numerical_param, output_param=output_param)
        k0 = Constant((np.pi, 0, 0))

        output = worm.solve(2, k0=k0, progress=True)
        self.assertTrue(output[0])
        FS = output[1]

        err = np.linalg.norm(FS.k[-1] - np.repeat(k0.values()[:, None], worm.N, axis = 1), axis=0).sum()
        self.assertAlmostEqual(0.0, err, places=2)


    def test_constant_control_2(self):


        pass



    # def test_solve(self):
    #
    #     worm = Worm()
    #
    #     A0, lam0 = 2*np.pi, 1.0
    #     q0 = 2*np.pi/lam0
    #
    #     k0 = Expression('A0*sin(q0*x[0]-2*pi*t)', degree=1, t=0.0, A0 = A0, q0=q0)
    #
    #     output = worm.solve(5, k0, progress=True, debug=True)
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
