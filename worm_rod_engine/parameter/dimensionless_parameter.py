# From built-in
from argparse import ArgumentParser
# From worm-rod-engine
from worm_rod_engine.parameter.physical_parameter import default_physical_parameter
from worm_rod_engine.parameter.util import convert_to_dimensionless


# Default physical parameter
default_dimensionless_parameter = convert_to_dimensionless(default_physical_parameter)
e = default_dimensionless_parameter.e
alpha = default_dimensionless_parameter.alpha
beta = default_dimensionless_parameter.beta
rho = default_dimensionless_parameter.rho
K_c = default_dimensionless_parameter.K_c
K_y = default_dimensionless_parameter.K_y
K_n = default_dimensionless_parameter.K_n

dimensionless_parameter_parser = ArgumentParser(description='dimensionless-parameter')

dimensionless_parameter_parser.add_argument('--e', type=float, default=e, help='Slenderness parameter')
dimensionless_parameter_parser.add_argument('--alpha', type=float, default=alpha, help='Relative external damping vs elastic response time')
dimensionless_parameter_parser.add_argument('--beta', type=float, default=beta, help='Relative internal damping vs elastic response time')
dimensionless_parameter_parser.add_argument('--rho', type=float, default=rho, help='Compliance factor')
dimensionless_parameter_parser.add_argument('--K_c', type=float, default=K_c, help='Linear drag coefficient ratio')
dimensionless_parameter_parser.add_argument('--K_y', type=float, default=K_y, help='Rotional drag coefficient ratio')
dimensionless_parameter_parser.add_argument('--K_n', type=float, default=K_n, help='Normal/lateral drag coefficient ratio')

