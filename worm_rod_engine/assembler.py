# From built-in
from typing import TYPE_CHECKING
from types import SimpleNamespace
import inspect
# From third-party
import numpy as np
from fenics import Function, project, Expression, Constant
# From worm-rod-engine
from worm_rod_engine.util import f2n, v2f
from worm_rod_engine.parameter.output_parameter import FUNCTION_KEYS

if TYPE_CHECKING:
    from worm_rod_engine.worm import Worm


class OutputAssembler():

    def __init__(self, worm: 'Worm', output_param: SimpleNamespace):
        self.worm = worm
        self.output_param = output_param
        self.cache = {}
        self.output = {}

    def update_state(self, r: Function, theta: Function, t: float):
        """
        Assemble output variables
        """
        self.cache['r'], self.cache['theta'], self.cache['t'] = r, theta, t


    def assemble(self, output_param_name: str):

        if output_param_name in self.cache:
            return self.cache[output_param_name]

        func = getattr(self.worm.PDE, output_param_name)

        # Extract the parameter names
        signature = inspect.signature(func)
        arg_names = [arg.name for arg in signature.parameters.values()]

        input_param = []

        for arg_name in arg_names:
            input_param.append(self.assemble(arg_name))

        self.cache[output_param_name] = func(*input_param)

        return self.cache[output_param_name]

    def assemble_input(self):
        """
        Assemble input variables
        """
        for input_name in ['eps0', 'k0']:
            if hasattr(self, input_name):
                v = getattr(self, input_name)
                if isinstance(v, Function):
                    v = f2n(v)
                elif isinstance(v, Expression):
                    v = f2n(v2f(v, fs=self.worm.PDE.function_spaces[input_name]))
                elif isinstance(v, Constant):
                    v = np.tile(v.values(), (1, self.worm.N))
                if isinstance(v, np.ndarray):
                    if self.worm.s_step is not None:
                        v = v[..., ::self.worm.s_step]
                self.output[input_name] = v

    def assemble_output(self):
        """
        Assemble output variables
        """
        for name, save in vars(self.output_param).items():
            if save:
                v = self.assemble(name)
                if isinstance(v, float):
                    pass
                elif isinstance(v, Function):
                    v = f2n(v)
                else:
                    fs = getattr(getattr(self.worm.PDE, name), 'function_space')
                    v = f2n(project(v, getattr(self.worm.PDE, fs)))

                if isinstance(v, np.ndarray):
                    if self.worm.s_step is not None:
                        v = v[..., ::self.worm.s_step]

                self.output[name] = v

