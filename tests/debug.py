# From built-in
import unittest
# From third-party
import numpy as np
from fenics import Expression
# From worm-rod-engine
from worm_rod_engine.worm import Worm
from worm_rod_engine.parameter.dimensionless_parameter import default_dimensionless_parameter
from worm_rod_engine.parameter.physical_parameter import default_physical_parameter
def debug_solve():

    worm = Worm()

    A0, lam0 = 2* np.pi, 1.0
    q0 = 2 * np.pi / lam0

    k0 = Expression('A0*sin(q0*x[0]-2*pi*t)', degree=1, t=0.0, A0=A0, q0=q0)

    output = worm.solve(5, k0, progress=True, debug=True)
    self.assertTrue(output[0], msg='Solve did not finsh')

def print_dimensionless_parameter():

    for key, value in vars(default_physical_parameter).items():
        if key in ['c_t', 'c_n']:
            value = value/default_physical_parameter.mu
        print(f'{key}={value}')

    for key, value in vars(default_dimensionless_parameter).items():
        print(f'{key}={value}')



if __name__ == '__main__':

    print_dimensionless_parameter()


    #debug_solve()
