# From built-in
from argparse import ArgumentParser, ArgumentTypeError
from types import SimpleNamespace
# From third-party
import numpy as np

def str2bool(v):
    """
    Handles boolean arguments
    """
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def resistive_force_theory(physical_param: SimpleNamespace):
    '''
    Calculates drag coefficients as predicted by resistive-force theory for incompressible Newtonian fluid environment
    '''
    mu, R, L0 = physical_param.mu, physical_param.R, physical_param.L0
    e = R / L0   # Slenderness parameter
    # Linear drag coefficients
    c_t = 2 * np.pi * mu / (np.log(1 / e) - 0.5)
    c_n = 4 * np.pi * mu / (np.log(1 / e) + 0.5)
    # Angular drag coefficients
    y_t = np.pi * mu * R ** 2
    y_n = 4 * np.pi * mu * R ** 2

    return c_t, c_n, y_t, y_n

def convert_to_dimensionless(physical_param: SimpleNamespace):
    """
    Converts physical to dimensionless parameters
    """
    pp = physical_param
    if pp.NIC:
        c_t, c_n, y_t, y_n = resistive_force_theory(physical_param)
    else:
        c_t, c_n, y_t, y_n = pp.c_t, pp.c_n, pp.y_t, pp.y_n

    I = 0.25 * np.pi*pp.R**4 # Second moment of area
    b = I*pp.E # Bending rigidity

    dimless_param = SimpleNamespace() # dimensional parameter
    dimless_param.e = pp.R / pp.L0
    dimless_param.alpha = c_t * pp.L0**4 / (b * pp.T)
    dimless_param.beta = pp.xi / pp.T
    dimless_param.rho = 0.5 / (1 + pp.nu)
    dimless_param.K_c = c_n / c_t
    dimless_param.K_y = y_n / y_t
    dimless_param.K_n= y_t / (c_t * pp.L0**2)

    for key, p in vars(dimless_param).items():
        assert p.check(['dimensionless']), f'{key} not dimensionless'
        setattr(dimless_param, key, p.magnitude)

    return dimless_param

class OutputArgumentParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_param_types = {}

    def add_argument(self, *args, out_type=None, **kwargs):
        arg =super().add_argument(*args, **kwargs)
        # Store the meta information if provided
        if out_type is not None:
            self.output_param_types[arg.dest] = out_type

    def get_output_type(self, argument_name):
        # Retrieve the meta information for a specific argument
        return self.output_param_types.get(argument_name)



