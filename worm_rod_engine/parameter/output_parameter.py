from argparse import ArgumentParser
import numpy as np
from worm_rod_engine.parameter.util import OutputArgumentParser, str2bool



FUNCTION_KEYS = ['r', 'theta', 'd1', 'd2', 'd3', 'k', 'eps', 'r_t', 'w', 'eps_t', 'k_t', 'f', 'N', 'F_M', 'l', 'M', 'L_M']

output_parameter_parser = OutputArgumentParser(description='output-parameter', allow_abbrev=False)

output_parameter_parser.add_argument('--t', type=str2bool, default=True, out_type=float, help='If True, save time stamp')
output_parameter_parser.add_argument('--r', type=str2bool, default=True, out_type=np.ndarray, help='If True, save centreline r')
output_parameter_parser.add_argument('--theta', type=str2bool, default=True, out_type=np.ndarray, help='If True, save Euler angles theta')
output_parameter_parser.add_argument('--d1', type=str2bool, default=False, out_type=np.ndarray, help='If True, save body-frame vectors')
output_parameter_parser.add_argument('--d2', type=str2bool, default=False, out_type=np.ndarray, help='If True, save body-frame vectors')
output_parameter_parser.add_argument('--d3', type=str2bool, default=False, out_type=np.ndarray, help='If True, save body-frame vectors')
output_parameter_parser.add_argument('--k', type=str2bool, default=False, out_type=np.ndarray, help='If True, save curvature vector')
output_parameter_parser.add_argument('--eps', type=str2bool, default=False, out_type=np.ndarray, help='If True, save state strain vector ')
output_parameter_parser.add_argument('--r_t', type=str2bool, default=False, out_type=np.ndarray, help='If True, save centreline velocity ')
output_parameter_parser.add_argument('--theta_t', type=str2bool, default=False, out_type=np.ndarray, help='If True, save centreline velocity ')
output_parameter_parser.add_argument('--w', type=str2bool, default=False, out_type=np.ndarray, help='If True, save angular velocity')
output_parameter_parser.add_argument('--eps_t', type=str2bool, default=False, out_type=np.ndarray, help='If True, save strain rate vector')
output_parameter_parser.add_argument('--k_t', type=str2bool, default=False, out_type=np.ndarray, help='If True, save curvature rate vector')
output_parameter_parser.add_argument('--f', type=str2bool, default=False, out_type=np.ndarray, help='If True, save external fluid force line density')
output_parameter_parser.add_argument('--N_', type=str2bool, default=False, out_type=np.ndarray, help='If True, save internal force resultant')
output_parameter_parser.add_argument('--F_M', type=str2bool, default=False, out_type=np.ndarray, help='If True, save muscle force')
output_parameter_parser.add_argument('--l', type=str2bool, default=False, out_type=np.ndarray, help='If True, save fluid torque density')
output_parameter_parser.add_argument('--M', type=str2bool, default=False, out_type=np.ndarray, help='If True, save interal torque resultant')
output_parameter_parser.add_argument('--L_M', type=str2bool, default=False, out_type=np.ndarray, help='If True, save muscle torque')
output_parameter_parser.add_argument('--DI_dot', type=str2bool, default=False, out_type=float, help='If True, save muscle torque')
output_parameter_parser.add_argument('--DE_dot', type=str2bool, default=False, out_type=float, help='If True, save muscle torque')
output_parameter_parser.add_argument('--V_dot', type=str2bool, default=False, out_type=float, help='If True, save muscle torque')
output_parameter_parser.add_argument('--W_dot', type=str2bool, default=False, out_type=float, help='If True, save muscle torque')

default_output_parameter = output_parameter_parser.parse_args([])
output_parameter_types = output_parameter_parser.output_param_types


