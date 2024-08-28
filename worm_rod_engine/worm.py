"""
Created on 12 May 2022

@author: lukas
"""
#Built-in imports
from typing import Dict, Optional, Union
from types import SimpleNamespace
import logging
from time import time
import warnings
# From third-party
import numpy as np
from fenics import *
from tqdm import tqdm
# From worm_rod_engine
from worm_rod_engine.util import v2f, f2n, count_decimal_places
from worm_rod_engine.frame import Frame, FrameSequence
from worm_rod_engine.assembler import OutputAssembler
from worm_rod_engine.pde import PDE, grad, finite_backwards_difference
from worm_rod_engine.parameter.dimensionless_parameter import default_dimensionless_parameter
from worm_rod_engine.parameter.numerical_parameter import default_numerical_parameter
from worm_rod_engine.parameter.output_parameter import default_output_parameter

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

if False:
    from fenics import dx

# Set Fenics LogLevel to Error to avoid logging to mess with progressbar
from dolfin import set_log_level, LogLevel
set_log_level(LogLevel.ERROR)

class Worm:

    def __init__(self,
        numerical_param: SimpleNamespace = default_numerical_parameter,
        dimensionless_param: SimpleNamespace = default_dimensionless_parameter,
        output_param: SimpleNamespace = default_output_parameter,
        fenics_solver_param: Dict = None):
        '''

        :param N: Number of mesh points
        :param dt: Dimensionless time step
        :param fdo: Finite difference order of time derivatives
        :param fet: Finite element type
        :param fed: Finite element degree
        :param fed: Finite element degree
        :param fenics_solver_param: kwargs dict passed to fenics.solve
        :param output_param: Specify if and what to save
        '''

        self.numerical_param = numerical_param
        self.N = numerical_param.N
        self.dt = numerical_param.dt
        self.N_report = numerical_param.N_report
        self.dt_report = numerical_param.dt_report
        self.fdo = numerical_param.fdo
        self.fet = numerical_param.fet
        self.fed = numerical_param.fed
        self.fenics_solver_param = {} if fenics_solver_param is None else fenics_solver_param
        self.dimensionless_param = dimensionless_param

        self._init_logger()
        self._init_function_space()
        self._initialize_report_intervals()

        self.assembler = OutputAssembler(self, output_param)

    def _init_logger(self):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s: %(name)s - %(message)s'))
        self.logger.addHandler(handler)

    def _initialize_report_intervals(self):

        # Initialize time step reporting interval
        if self.dt_report is not None:
            assert self.dt_report > self.dt, \
                f"dt_report={self.dt_report} must be larger than dt={self.dt}."
            self.t_step = int(round(self.dt_report / self.dt))
        else:
            self.t_step = None

        if self.N_report is not None:
            assert self.N_report < self.N, \
                f"N_report={self.N_report} must be less than N={self.N}."
            self.s_step = int(round(self.N / self.N_report))
        else:
            self.s_step = None

        return

    def _init_input(self, eps0: Union[Constant, Expression, None], k0: Union[Constant, Expression, None]):

        if isinstance(eps0, (Constant, Expression,)) or eps0 is None:
            self.eps0 = eps0
        elif isinstance(eps0, np.ndarray) or eps0 == np.ndarray:
            self.eps0 = Function(self.V3)

        if isinstance(k0, (Constant, Expression)) or k0 is None:
            self.k0 = k0
        elif isinstance(eps0, np.ndarray) or eps0 == np.ndarray:
            self.k0 = Function(self.V3)

        self.assembler.cache['eps0'], self.assembler.cache['k0'] = self.eps0, self.k0

    def _init_function_space(self):
        '''
        Initialise finite element function spaces
        '''
        self.mesh = UnitIntervalMesh(self.N - 1)
        # Spatial element
        self.dx = Measure('dx', domain=self.mesh)
        # Finite elements for 1 dimensional spatial coordinate s
        P1 = FiniteElement(self.fet, self.mesh.ufl_cell(), self.fed)
        # State variables r and theta are 3 dimensional vector-valued functions of s
        P1_3 = MixedElement([P1] * 3)
        # Function space for scalar functions of s
        self.V = FunctionSpace(self.mesh, P1)
        # Function space for 3 component vector-valued functions of s
        self.V3 = FunctionSpace(self.mesh, P1_3)
        # Trial function space for 6 component vector-valued function composed of r and theta
        self.W = FunctionSpace(self.mesh, MixedElement(P1_3, P1_3))

    def _init_form(self):
        """
        Weak form of PDE
        """
        u = TrialFunction(self.W)
        phi = TestFunction(self.W)

        r, theta = split(u)
        phi_r, phi_theta = split(phi)

        r_old_arr = [split(u)[0] for u in self.u_old_arr]
        theta_old_arr = [split(u)[1] for u in self.u_old_arr]

        # First time derivatives
        r_t = finite_backwards_difference(1, self.fdo, r, r_old_arr, self.dt)
        theta_t = finite_backwards_difference(1, self.fdo, theta, theta_old_arr, self.dt)

        # Head functions are approximated by previous time/iteration step to linearize the equations of motion
        self.u_h = Function(self.W)
        r_h, theta_h = split(self.u_h)

        self.PDE = PDE(self)
        # Head expressions are linear in unkowns
        Q_h = self.PDE.Q(theta_h)
        A_h = self.PDE.A(theta_h)
        A_h_t = self.PDE.A_t(theta_h, theta_t)
        T_h = self.PDE.T(r_h)

        # Angular velocity vector
        w = self.PDE.w(A_h, theta_t)
        # Strain vector
        eps = self.PDE.eps(Q_h, r)
        # Generalized curvature vector
        k = self.PDE.k(A_h, theta)
        # Strain rate vector
        eps_t = self.PDE.eps_t(Q_h, r_h, r_t, w)
        # Curvature rate vector
        k_t = self.PDE.k_t(A_h, A_h_t, theta_h, theta_t)
        # Internal force
        N = self.PDE.N(Q_h, eps, eps_t)

        # Internal force and muscle force
        if self.eps0 is None:
            N_and_F_M = N
        else:
            N_and_F_M = self.PDE.N_and_F_M(Q_h, eps, eps_t)

        # Internal torque and muscle torque
        if self.k0 is None:
            M_and_L_M = self.PDE.M(Q_h, k, k_t)
        else:
            M_and_L_M = self.PDE.M_and_L_M(Q_h, k, k_t)

        # External fluid drag torque
        l = self.PDE.l(Q_h, w)
        # External fluid drag force
        f = self.PDE.f(Q_h, r_t)
        # linear balance
        eq1 = dot(f, phi_r) * dx - dot(N_and_F_M, grad(phi_r)) * dx
        # Angular balance
        eq2 = (
                dot(l, phi_theta) * dx
                + dot(T_h * N, phi_theta) * dx
                - dot(M_and_L_M, grad(phi_theta)) * dx
        )

        equation = eq1 + eq2

        self.F_op, self.L = lhs(equation), rhs(equation)

        return

    def _assign_initial_values(self, F0: Optional[Frame] = None):
        '''
        Initialise initial state and state history
        '''
        # Trial functions
        r0 = Function(self.V3)
        theta0 = Function(self.V3)

        # If no frame is given, use default
        if F0 is None:
            # Set initial configuration
            r0.assign(Expression(("0", "0", "x[0]"), degree=self.fed))
            theta0.assign(Expression(("0", "0", "0"), degree=self.fed))
            self.t = 0.0
        # If Numpy frame is given, assign array values to fenics functions
        else:
            v2f(F0.r, r0)
            v2f(F0.theta, theta0)
            self.t = F0.t

        # Set past states to initial state
        self.u_old_arr = [Function(self.W) for _ in np.arange(self.fdo)]

        # Assign (r, theta) tuple in [V3, V3] to u in W
        fa = FunctionAssigner(self.W, [self.V3, self.V3])

        for u_old_n in self.u_old_arr:
            fa.assign(u_old_n, [r0, theta0])

        return

    def _update_inputs(self, eps0: Optional[np.ndarray] = None, k0: Optional[np.ndarray] = None):
        """
        Update input curvature and input strain vector
        """
        if eps0 is not None:
            v2f(eps0, self.eps0)
        elif isinstance(self.eps0, Expression):
            if hasattr(self.eps0, 't'):
                self.eps0.t = self.t
        if k0 is not None:
            v2f(k0, self.k0)
        if isinstance(self.k0, Expression):
            if hasattr(self.k0, 't'):
                self.k0.t = self.t

    def _picard_iteration(self):

            """Solve nonlinear system of equations using picard iteration"""

            # Trial function
            u = Function(self.W)

            # Solution from previous time step
            u_old = self.u_old_arr[-1]
            r_old, theta_old = u_old.split()
            r_old_arr = r_old.compute_vertex_values(self.mesh).reshape(3, self.N)
            theta_old_arr = theta_old.compute_vertex_values(self.mesh).reshape(3, self.N)

            # Initial guess
            self.u_h.assign(u_old)

            tol = self.picard['tol']
            lr = self.picard['lr']
            maxiter = self.picard['max_iter']

            i = 0
            converged = False

            while i < maxiter:
                solve(self.F_op == self.L, u, solver_parameters=self.fenics_solver_param)
                r, theta = u.split()
                r_h, theta_h = self.u_h.split()

                # Error
                err_r = assemble(sqrt((r-r_h)**2)*dx)
                err_theta = assemble(sqrt((theta-theta_h)**2)*dx)

                # Normalize by average change per time step
                norm_r = assemble(sqrt((r - r_old)**2)*dx)
                norm_theta = assemble(sqrt((theta - theta_old)**2)*dx)

                rel_err_r  = err_r / max(norm_r, 1.0e-12)
                rel_err_theta  = err_theta / max(norm_theta, 1.0e-12)

                if rel_err_r < tol and rel_err_theta < tol:
                    if not self.quiet:
                        print(
                            f"Picard iteration converged after {i} iterations: err_r={err_r}, err_theta={err_theta}"
                        )
                    converged = True
                    break

                self.u_h.assign(lr * u + (1.0 - lr) * self.u_h)
                i += 1

            assert converged, 'Picard iteration did not converge'

            return u

#================================================================================================
# class API
#================================================================================================

    def initialise(self,
        eps0: Optional[Union[Constant, Expression, np.ndarray]] = None,
        k0: Optional[Union[Constant, Expression, np.ndarray]] = None,
        F0: Optional[Frame] = None):
        """
        Initialise worm object for given model parameters, control sequence (optional) and initial frame (optional).
        """
        self._assign_initial_values(F0)
        self._init_input(eps0, k0)
        self._init_form()

    def solve(self,
        T_sim: float,
        eps0: Optional[Union[Constant, Expression]] = None,
        k0: Optional[Union[Constant, Expression]] = None,
        F0: Optional[Frame] = None,
        progress: bool = False,
        log: bool = False,
        debug: bool = False):
        """
        Run the forward model for 'T_sim' dimensionless time units.
        """
        self.initialise(eps0, k0, F0)

        # Number of time steps
        self.n_t_step = int(T_sim / self.dt)
        dt_decimal_places= count_decimal_places(self.dt)

        if not log:
            self.logger.setLevel(logging.CRITICAL)
        if progress:
            pbar = tqdm(total = self.n_t_step)

        # Frames
        frames = []
        Frame(**self.assembler.output)

        start_time = time()

        self.logger.info(f'Solve forward (t={self.t:.{dt_decimal_places}f}..{self.t + T_sim:.{dt_decimal_places}f}) / n_steps={self.n_t_step}')
        # Try block allows for customized exception handling. If we run simulations in parallel, we don't want the
        # whole batch to crash if individual simulations fail
        try:
            for i in range(self.n_t_step):

                eps0_i = eps0[i, :] if isinstance(eps0, np.ndarray) else None
                k0_i = k0[i, :] if isinstance(k0, np.ndarray) else None

                assemble = (i + 1) % self.t_step == 0 if self.t_step is not None else True

                self.update_state(eps0_i, k0_i, assemble=assemble)

                if assemble:
                    frames.append(Frame(**self.assembler.output))

                if progress:
                    pbar.update(1)

        except Exception as e:
            if debug: raise e
            # If simulation fails, return simulation outputs til now and exception upstream
            self.logger.error(f'Solver failed at time step {i} out of n_steps={self.n_t_step}, abort simulation')
            return False, FrameSequence(frames), e

        sim_time = time() - start_time

        return True, FrameSequence(frames), sim_time

    def update_state(self,
        eps0: Optional[np.ndarray] = None,
        k0:  Optional[np.ndarray] = None,
        assemble: bool = False):
        '''
        Solve time step and save solution to Frame
        '''
        self.t += self.dt
        self._update_inputs(eps0, k0)
        # Update initial guess of head functions with solution from last time step
        self.u_h.assign(self.u_old_arr[-1])

        if self.numerical_param.pic_on:
            u = self._picard_iteration()
        else:
            u = Function(self.W)
            solve(self.F_op == self.L, u, solver_parameters = self.fenics_solver_param)

        assert not np.isnan(u.vector().get_local()).any(), f'Solution at t={self._t:.{self.D}f} contains nans!'

        # Frame and outputs need to be assembled before u_old_arr is updated for derivatives to use correct data points
        if assemble:
            r, theta = u.split(deepcopy=True)
            self.assembler.update_state(r, theta, self.t)
            self.assembler.assemble_output()
            self.assembler.clear_cache()

        # update past solution cache
        for n, u_n in enumerate(self.u_old_arr[:-1]):
            u_n.assign(self.u_old_arr[n + 1])

        self.u_old_arr[-1].assign(u)

        return


