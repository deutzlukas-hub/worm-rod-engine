from argparse import ArgumentParser
from worm_rod_engine.parameter.util import str2bool

numerical_argument_parser = ArgumentParser(description='solver-parameter', allow_abbrev=False)

# Discretization
numerical_argument_parser.add_argument('--N', type=int, default=750, help ='Number of mesh points')
numerical_argument_parser.add_argument('--dt', type=float, default=0.001, help ='Dimensionless time step')
numerical_argument_parser.add_argument('--N_report', type=lambda x: None if x.lower() == 'none' else float(x), default=None, help='Save only N_report mesh points from the finer grid simulation.')
numerical_argument_parser.add_argument('--dt_report', type=lambda x: None if x.lower() == 'none' else float(x), default=None, help='Save data at intervals of dt_report time steps.')
numerical_argument_parser.add_argument('--fdo', type=int, default=2, help='Order of finite difference approximation of first time derivatives')
numerical_argument_parser.add_argument('--fet', type=str, default='Lagrange', help="Type of finite element.")
numerical_argument_parser.add_argument('--fed', type=int, default=1, help="Degree of the finite element.")
# Pircard iteration
numerical_argument_parser.add_argument('--pic_on', type=str2bool, default=False, help='If true, solve nonlinear pde with picard iteration')
numerical_argument_parser.add_argument('--pic_max_iter', type=int, default=100, help='Maximum number of iteration steps')
numerical_argument_parser.add_argument('--pic_lr', type=float, default=0.5, help='Learning rate')
numerical_argument_parser.add_argument('--pic_tol', type=float, default=1e-2, help='Error tolerance')

default_numerical_parameter = numerical_argument_parser.parse_args()
