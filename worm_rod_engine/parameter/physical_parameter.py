# From build-in
from argparse import ArgumentParser
# From third-party
import pint
# from worm-rod-engine
from worm_rod_engine.parameter.util import str2bool, resistive_force_theory

ureg = pint.UnitRegistry()

phyisical_parameter_parser = ArgumentParser(description='celegans-parameter', allow_abbrev=False)

phyisical_parameter_parser.add_argument('--physical_to_dimensionless', type=str2bool, default=False,
                                        help='If True, calculate dimensionless parameters from physical')
# Geometric parameter
phyisical_parameter_parser.add_argument('--L0', type=float, default=1130 * 1e-6, help='Natural worm length [meter]')
phyisical_parameter_parser.add_argument('--R', type=float, default=32 * 1e-6, help='Maximum worm radius in [meter]')
# Material parameter
phyisical_parameter_parser.add_argument('--E', type=float, default=1.21e5, help="Young's modulus [pascal]")
phyisical_parameter_parser.add_argument('--xi', type=float, default=10**(-1.7), help="Relative damping coefficient [second]")
phyisical_parameter_parser.add_argument('--nu', type=float, default=0.5, help="Poisson ratio")
# Undulation period
phyisical_parameter_parser.add_argument('--T', type=float, default=1.0, help='Characteristic time scale [second]')
# Environment parameter
phyisical_parameter_parser.add_argument('--NIC', type=str2bool, default=True, help='Specify if the fluid is Newtonian incompressible')
phyisical_parameter_parser.add_argument('--mu', type=float, default=1e-3, help='Fluid viscosity [pascal*second]')

phyisical_parameter = phyisical_parameter_parser.parse_args([])
c_t, c_n, y_t, y_n = resistive_force_theory(phyisical_parameter)

phyisical_parameter_parser.add_argument('--c_t', type=float, default=c_t, help='Tangential/longitudinal linear drag coefficient [pascal*second]')
phyisical_parameter_parser.add_argument('--c_n', type=float, default=c_n, help='Normal/lateral linear drag coefficient [pascal*second]')
phyisical_parameter_parser.add_argument('--y_t', type=float, default=y_t, help='Tangential/longitudinal angular drag coefficient [pascal*second*meter^2]')
phyisical_parameter_parser.add_argument('--y_n', type=float, default=y_n, help='Normal/lateral angular drag coefficient [pascal*second*meter^2]')

default_physical_parameter = phyisical_parameter_parser.parse_args([])





