from argparse import ArgumentParser, BooleanOptionalAction
from worm_rod_engine.parameter.util import OutputArgumentParser
import numpy as np

FUNCTION_KEYS = ['r', 'theta', 'd1', 'd2', 'd3', 'k', 'eps', 'r_t', 'w', 'eps_t', 'k_t', 'f', 'N', 'F_M', 'l', 'M', 'L_M']

output_parameter_parser = OutputArgumentParser(description='output-parameter')

output_parameter_parser.add_argument('--t', action=BooleanOptionalAction, default=True, out_type=float, help='If True, save time stamp')
output_parameter_parser.add_argument('--r', action=BooleanOptionalAction, default=True, out_type=np.ndarray, help='If True, save centreline r')
output_parameter_parser.add_argument('--theta', action=BooleanOptionalAction, default=True, out_type=np.ndarray, help='If True, save Euler angles theta')
output_parameter_parser.add_argument('--d1', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save body-frame vectors')
output_parameter_parser.add_argument('--d2', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save body-frame vectors')
output_parameter_parser.add_argument('--d3', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save body-frame vectors')
output_parameter_parser.add_argument('--k', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save curvature vector')
output_parameter_parser.add_argument('--eps', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save state strain vector ')
output_parameter_parser.add_argument('--r_t', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save centreline velocity ')
output_parameter_parser.add_argument('--w', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save angular velocity')
output_parameter_parser.add_argument('--eps_t', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save strain rate vector')
output_parameter_parser.add_argument('--k_t', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save curvature rate vector')
output_parameter_parser.add_argument('--f', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save external fluid force line density')
output_parameter_parser.add_argument('--N', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save internal force resultant')
output_parameter_parser.add_argument('--F_M', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save muscle force')
output_parameter_parser.add_argument('--l', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save fluid torque density')
output_parameter_parser.add_argument('--M', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save interal torque resultant')
output_parameter_parser.add_argument('--L_M', action=BooleanOptionalAction, default=False, out_type=np.ndarray, help='If True, save muscle torque')
output_parameter_parser.add_argument('--DI_dot', action=BooleanOptionalAction, default=False, out_type=float, help='If True, save muscle torque')
output_parameter_parser.add_argument('--DE_dot', action=BooleanOptionalAction, default=False, out_type=float, help='If True, save muscle torque')
output_parameter_parser.add_argument('--V_dot', action=BooleanOptionalAction, default=False, out_type=float, help='If True, save muscle torque')
output_parameter_parser.add_argument('--W_dot', action=BooleanOptionalAction, default=False, out_type=float, help='If True, save muscle torque')



default_output_parameter = output_parameter_parser.parse_args()
output_parameter_types = output_parameter_parser.output_param_types


