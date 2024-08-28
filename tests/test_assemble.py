# From built-in
import unittest
import warnings
# From third-party
import numpy as np
from fenics import Expression, Constant
# From worm-rod-engine
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.parameter.output_parameter import output_parameter_parser
from worm_rod_engine.worm import Worm
from worm_rod_engine.util import v2f, f2n

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class TestAssembler(unittest.TestCase):

    def test_assemble_random(self):

        output_param = output_parameter_parser.parse_args(['--eps0', '--k0'])
        N = np.random.randint(100, 1000)
        numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
        worm = Worm(numerical_param=numerical_param, output_param=output_param)
        worm.initialise(eps0=np.ndarray, k0=np.ndarray)

        r_arr_in = np.random.rand(3, N)
        theta_arr_in = np.random.rand(3, N)
        eps0_arr_in = np.random.rand(3, N)
        k0_arr_in = np.random.rand(3, N)
        t_in = np.random.rand()

        r = v2f(r_arr_in, fs=worm.V3)
        theta = v2f(theta_arr_in, fs=worm.V3)

        worm._update_inputs(eps0_arr_in, k0_arr_in)
        worm.assembler.update_state(r, theta, t_in)
        worm.assembler.assemble_output()

        err_r = np.linalg.norm(r_arr_in - worm.assembler.output['r'], axis=0).sum()
        err_theta = np.linalg.norm(theta_arr_in - worm.assembler.output['theta'], axis=0).sum()
        err_eps0 = np.linalg.norm(eps0_arr_in - worm.assembler.output['eps0'], axis=0).sum()
        err_k0 = np.linalg.norm(k0_arr_in - worm.assembler.output['k0'], axis=0).sum()
        err_t = t_in - worm.assembler.output['t']
        self.assertAlmostEqual(err_r, 0.0)
        self.assertAlmostEqual(err_theta, 0.0)
        self.assertAlmostEqual(err_eps0, 0.0)
        self.assertAlmostEqual(err_k0, 0.0)
        self.assertAlmostEqual(err_t, 0.0)
    def test_assemble_expression(self):

        for _ in range(2):

            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            output_param = output_parameter_parser.parse_args(['--eps0', '--k0'])
            worm = Worm(numerical_param=numerical_param, output_param=output_param)
            s_arr = np.linspace(0, 1, N)

            eps0 = Expression(('cos(x[0]+t)', 'sin(x[0]+t)', '0'), degree=worm.fed, t=0.0)
            k0 = Expression(('sin(x[0]+t)', 'cos(x[0]+t)', '0'), degree=worm.fed, t=0.0)
            worm.initialise(eps0, k0)

            for t in np.arange(0.01, 1, 0.01):

                r = v2f(np.random.rand(3, N), fs=worm.V3)
                theta = v2f(np.random.rand(3, N), fs=worm.V3)

                eps0.t =  t
                k0.t = t

                eps0_arr_in = np.vstack((np.cos(s_arr + t), np.sin(s_arr + t), np.zeros(N)))
                k0_arr_in = np.vstack((np.sin(s_arr + t), np.cos(s_arr + t), np.zeros(N)))

                worm.assembler.update_state(r, theta, t)
                worm.assembler.assemble_output()

                err_eps0 = np.linalg.norm(eps0_arr_in - worm.assembler.output['eps0'], axis=0).sum()
                err_k0 = np.linalg.norm(k0_arr_in - worm.assembler.output['k0'], axis=0).sum()
                self.assertAlmostEqual(err_eps0, 0.0)
                self.assertAlmostEqual(err_k0, 0.0)

    def test_assemble_constant(self):

        for _ in range(2):

            N = np.random.randint(100, 1000)
            numerical_param = numerical_argument_parser.parse_args(['--N', str(N)])
            output_param = output_parameter_parser.parse_args(['--eps0', '--k0'])
            worm = Worm(numerical_param=numerical_param, output_param=output_param)
            eps0 = Constant(np.random.rand(3))
            k0 = Constant(np.random.rand(3))

            worm.initialise(eps0, k0)

            eps0_arr_in = np.tile(eps0.values(), (1, worm.N))
            k0_arr_in = np.tile(k0.values(), (1, worm.N))

            for t in np.arange(0, 1.0, 0.1):

                r = v2f(np.random.rand(3, N), fs=worm.V3)
                theta = v2f(np.random.rand(3, N), fs=worm.V3)

                worm._update_inputs()
                worm.assembler.update_state(r, theta, t)
                worm.assembler.assemble_output()

                err_eps0 = np.linalg.norm(eps0_arr_in - worm.assembler.output['eps0'], axis=0).sum()
                err_k0 = np.linalg.norm(k0_arr_in - worm.assembler.output['k0'], axis=0).sum()
                self.assertAlmostEqual(err_eps0, 0.0)
                self.assertAlmostEqual(err_k0, 0.0)

if __name__ == '__main__':

    unittest.main()










