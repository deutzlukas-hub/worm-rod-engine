# From built-in
from typing import Union, TYPE_CHECKING
from types import SimpleNamespace
import inspect
# From third-party
import numpy as np
from fenics import Function, project, Expression, Constant
# From worm-rod-engine
from worm_rod_engine.pde import PDE
from worm_rod_engine.util import f2n, v2f

if TYPE_CHECKING:
    from worm_rod_engine.worm import Worm

class OutputAssembler():

    def __init__(self, worm: 'Worm', output_param: SimpleNamespace):
        self.worm = worm
        self.output_param = output_param
        self.cache = {}
        self.output = {}

    def clear_cache(self):
        for key in list(self.cache):
            if key not in ['k0', 'eps0']:
                del self.cache[key]

    def update_state(self,
        r: Function,
        theta: Function,
        t: float):
        """
        Assemble output variables
        """
        self.cache['r'] = r
        self.cache['theta'] = theta
        self.cache['t'] = t


    def assemble(self, output_param_name: str):

        if output_param_name in self.cache:
            return self.cache[output_param_name]

        func = getattr(PDE, output_param_name)

        # Extract the parameter names
        signature = inspect.signature(func)
        arg_names = [arg.name for arg in signature.parameters.values()]

        input_param = []

        for arg_name in arg_names:
            input_param.append(self.assemble(arg_name))

        self.cache[output_param_name] = func(*input_param)

        return self.cache[output_param_name]

    def assemble_output(self):
        """
        Assemble output variables
        """
        for param_name, save in vars(self.output_param).items():
            if save:
                v = self.assemble(param_name)

                if isinstance(v, Function):
                    v = f2n(v)
                elif isinstance(v, float):
                    pass
                elif isinstance(v, Expression):
                    v = f2n(v2f(v, fs=self.worm.V3))
                elif isinstance(v, Constant):
                    v = np.tile(v.values(), (1, self.worm.N))
                else:
                    v = f2n(project(v, self.worm.V3))

                if isinstance(v, np.ndarray):
                    if self.worm.s_step is not None:
                        v = v[..., ::self.worm.s_step]
                self.output[param_name] = v

