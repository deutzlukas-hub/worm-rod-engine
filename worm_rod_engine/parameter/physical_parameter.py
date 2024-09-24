# From build-in
from argparse import ArgumentParser
# From third-party
import pint
# from worm-rod-engine
from worm_rod_engine.parameter.util import str2bool, resistive_force_theory

ureg = pint.UnitRegistry()

phyisical_parameter_parser = ArgumentParser(description='celegans-parameter')

# Geometric parameter
phyisical_parameter_parser.add_argument('--L0', type=lambda s: float(s) * ureg.meter,
                                        default=1130 * 1e-6 * ureg.meter, help='Natural worm length')
phyisical_parameter_parser.add_argument('--R', type=lambda s: float(s) * ureg.meter,
                                        default=32 * 1e-6 * ureg.meter, help='Maximum worm radius')
# Material parameter
phyisical_parameter_parser.add_argument('--E', type=lambda s: float(s) * ureg.pascal,
                                        default=1.21e5 * ureg.pascal, help="Young's modulus")
phyisical_parameter_parser.add_argument('--xi', type=lambda s: float(s) * ureg.second,
                                        default=10**(-1.7)*ureg.second, help="Relative damping coefficient")
phyisical_parameter_parser.add_argument('--nu', type=lambda s: float(s) * ureg.dimensionless,
                                        default=0.5 * ureg.dimensionless, help="Poisson ratio")
# Undulation period
phyisical_parameter_parser.add_argument('--T', type=lambda s: float(s) * ureg.second,
                                        default=1.0 * ureg.second, help='Characteristic time scale')
# Environment parameter
phyisical_parameter_parser.add_argument('--NIC', type=str2bool, default=True,
                                        help='Specify if the fluid is Newtonian incompressible')
phyisical_parameter_parser.add_argument('--mu', type=lambda s: float(s) * ureg.pascal * ureg.second,
                                        default=1e-3 * ureg.pascal * ureg.second, help='Fluid viscosity')

phyisical_parameter = phyisical_parameter_parser.parse_args([])
c_t, c_n, y_t, y_n = resistive_force_theory(phyisical_parameter)

phyisical_parameter_parser.add_argument('--c_t', type=lambda s: float(s) * ureg.pascal * ureg.second,
                                        default=c_t, help='Tangential/longitudinal linear drag coefficient')
phyisical_parameter_parser.add_argument('--c_n', type=lambda s: float(s) * ureg.pascal * ureg.second,
                                        default=c_n, help='Normal/lateral linear drag coefficient')
phyisical_parameter_parser.add_argument('--y_t', type=lambda s: float(s) * ureg.pascal * ureg.second,
                                        default=y_t, help='Tangential/longitudinal angular drag coefficient')
phyisical_parameter_parser.add_argument('--y_n', type=lambda s: float(s) * ureg.pascal * ureg.second,
                                        default=y_n, help='Normal/lateral angular drag coefficient')

default_physical_parameter = phyisical_parameter_parser.parse_args([])





