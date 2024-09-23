# From built-in
from abc import ABC, abstractmethod
from typing import Union, Optional, TYPE_CHECKING
# From third-party
import numpy as np
from fenics import *
# From worm-rod-engine
from worm_rod_engine.util import v2f
from worm_rod_engine.frame import Frame
from worm_rod_engine.pde.util import finite_backwards_difference

if TYPE_CHECKING:
    from worm_rod_engine.worm import Worm

class PDE_Cosserat(ABC):

    def __init__(self, worm: 'Worm'):

        self.worm = worm

        self._init_constants()
        self._init_function_space()


    @abstractmethod
    def r0_default(self):
        pass

    @abstractmethod
    def theta0_default(self):
        pass

    @abstractmethod
    def _init_S(self):
        pass

    @abstractmethod
    def _init_S_tilde(self):
        pass

    @abstractmethod
    def _init_B(self):
        pass

    @abstractmethod
    def _init_B_tilde(self):
        pass

    def _init_constants(self):

        self.e = self.worm.dimensionless_param.e
        self.alpha = self.worm.dimensionless_param.alpha
        self.beta = self.worm.dimensionless_param.beta
        self.rho = self.worm.dimensionless_param.rho
        self.K_c = self.worm.dimensionless_param.K_c
        self.K_n = self.worm.dimensionless_param.K_n
        self.K_y = self.worm.dimensionless_param.K_y

        self.S = self._init_S()
        self.S_tilde = self._init_S_tilde()
        self.B = self._init_B()
        self.B_tilde = self._init_B_tilde()

        return

    @abstractmethod
    def _init_function_space(self):
        pass

    def _init_form(self):
        pass

    def _init_input(self,
        eps0: Union[Constant, Expression, None],
        k0: Union[Constant, Expression, None],
        asseble_input: bool):

        if isinstance(eps0, (Constant, Expression,)) or eps0 is None:
            self.eps0 = eps0
        elif isinstance(eps0, np.ndarray) or eps0 == np.ndarray:
            self.eps0 = Function(self.function_spaces['eps0'])

        if isinstance(k0, (Constant, Expression)) or k0 is None:
            self.k0 = k0
        elif isinstance(eps0, np.ndarray) or eps0 == np.ndarray:
            self.k0 = Function(self.function_spaces['k0'])

        if asseble_input:
            self.worm.assembler.eps0, self.worm.assembler.k0 = self.eps0, self.k0

    def _assign_initial_values(self, F0: Optional[Frame] = None):
        '''
        Initialise initial state and state history
        '''
        # Trial functions
        r0 = Function(self.function_spaces['r'])
        theta0 = Function(self.function_spaces['theta'])

        # If no frame is given, use default
        if F0 is None:
            # Set initial configuration
            r0.assign(self.r0_default())
            theta0.assign(self.theta0_default())
        # If Numpy frame is given, assign array values to fenics functions
        else:
            v2f(F0.r, r0)
            v2f(F0.theta, theta0)

        # Assign (r, theta) tuple in [V2, V] to u in W
        fa = FunctionAssigner(self.W, [self.function_spaces['r'], self.function_spaces['theta']])

        for u_old_n in self.u_old_arr:
            fa.assign(u_old_n, [r0, theta0])

        return

    @staticmethod
    @abstractmethod
    def Q():
        pass

    @staticmethod
    @abstractmethod
    def eps():
        pass

    @staticmethod
    @abstractmethod
    def k():
        pass

    @staticmethod
    @abstractmethod
    def eps_t(self):
        pass


    @staticmethod
    @abstractmethod
    def k_t(self):
        pass

    @abstractmethod
    def f(self):
        pass

    @abstractmethod
    def l(self):
        pass

    @abstractmethod
    def N_(self):
        pass

    @abstractmethod
    def M(self):
        pass

    def N_and_F_M(self):
        pass

    @abstractmethod
    def M_and_L_M(self):
        pass

















