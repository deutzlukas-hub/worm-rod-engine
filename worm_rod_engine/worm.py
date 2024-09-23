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
from dolfin.cpp.function import Constant as cppConstant
from tqdm import tqdm
# From worm_rod_engine
from worm_rod_engine.util import v2f, count_decimal_places
from worm_rod_engine.frame import Frame, FrameSequence
from worm_rod_engine.assembler import OutputAssembler
from worm_rod_engine.parameter.dimensionless_parameter import default_dimensionless_parameter
from worm_rod_engine.parameter.numerical_parameter import default_numerical_parameter
from worm_rod_engine.parameter.output_parameter import default_output_parameter
from worm_rod_engine.pde.cosserat_rod_2D import CosseratRod2D
from worm_rod_engine.pde.cosserat_rod_3D import CosseratRod3D

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

if False:
    from fenics import dx

# Set Fenics LogLevel to Error to avoid logging to mess with progressbar
from dolfin import set_log_level, LogLevel
set_log_level(LogLevel.ERROR)

INPUT_KEYS = ['eps0', 'k0']

class Worm:

    def __init__(self,
        dimension=3,
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
        self.dimensionless_param = dimensionless_param
        self.N = numerical_param.N
        self.dt = numerical_param.dt
        self.N_report = numerical_param.N_report
        self.dt_report = numerical_param.dt_report

        if dimension == 2:
            self.PDE = CosseratRod2D(self)
        elif dimension == 3:
            self.PDE = CosseratRod3D(self)
        else:
            assert False, "'dimension' must be in [2, 3]"

        self._init_logger()
        self._initialize_report_intervals()
        self.assembler = OutputAssembler(self, output_param)

        self.fenics_solver_param = {} if fenics_solver_param is None else fenics_solver_param

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

    def _update_inputs(self, eps0: Optional[np.ndarray] = None, k0: Optional[np.ndarray] = None):
        """
        Update input curvature and input strain vector
        """
        if eps0 is not None:
            v2f(eps0, self.PDE.eps0)
        elif isinstance(self.PDE.eps0, Expression):
            if hasattr(self.PDE.eps0, 't'):
                if isinstance(self.PDE.eps0.t, float):
                    self.PDE.eps0.t = self.t
                if isinstance(self.PDE.eps0.t, cppConstant):
                    self.PDE.eps0.t.assign(self.t)
        if k0 is not None:
            v2f(k0, self.PDE.k0)
        if isinstance(self.PDE.k0, Expression):
            if hasattr(self.PDE.k0, 't'):
                if isinstance(self.PDE.k0.t, float):
                    self.PDE.k0.t = self.t
                if isinstance(self.PDE.k0.t, cppConstant):
                    self.PDE.k0.t.assign(self.t)



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
        F0: Optional[Frame] = None,
        assemble_input: bool = True):
        """
        Initialise worm object for given model parameters, control sequence (optional) and initial frame (optional).
        """
        self.t = 0.0 if F0 is None else F0.t
        self.PDE._init_input(eps0, k0, assemble_input)
        self.PDE._init_form()
        self.PDE._assign_initial_values(F0)

    def solve(self,
        T_sim: float,
        eps0: Optional[Union[Constant, Expression]] = None,
        k0: Optional[Union[Constant, Expression]] = None,
        F0: Optional[Frame] = None,
        assemble_input: bool = True,
        progress: bool = True,
        log: bool = False,
        debug: bool = False,
        pbar: Optional[tqdm] = None):
        """
        Run the forward model for 'T_sim' dimensionless time units.
        """
        self.initialise(eps0, k0, F0, assemble_input)

        # Number of time steps
        self.n_t_step = int(T_sim / self.dt)
        self.dt_dp = count_decimal_places(self.dt)

        if not log:
            self.logger.setLevel(logging.CRITICAL)
        if progress:
            if pbar is None:
                pbar = tqdm(total = self.n_t_step)
            else:
                pbar.total = self.n_t_step

        # Frames
        frames = []

        start_time = time()

        self.logger.info(f'Solve forward (t={self.t:.{self.dt_dp}f}..{self.t + T_sim:.{self.dt_dp}f}) / n_steps={self.n_t_step}')
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
            runtime = time() - start_time
            # If simulation fails, return simulation outputs til now and exception upstream
            self.logger.error(f'Solver failed at time step {i} out of n_steps={self.n_t_step}, abort simulation')
            return {'exit_status': False, 'FS': FrameSequence(frames), 'runtime': runtime}, e

        runtime = time() - start_time
        return {'exit_status': True, 'FS': FrameSequence(frames), 'runtime': runtime}, None


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
        self.PDE.u_h.assign(self.PDE.u_old_arr[-1])

        if self.numerical_param.pic_on:
            u = self._picard_iteration()
        else:
            u = Function(self.PDE.W)
            solve(self.PDE.F_op == self.PDE.L, u, solver_parameters = self.fenics_solver_param)

        assert not np.isnan(u.vector().get_local()).any(), f'Solution at t={self._t:.{self.dt_dp}f} contains nans!'

        # Frame and outputs need to be assembled before u_old_arr is updated for derivatives to use correct data points
        if assemble:
            r, theta = u.split(deepcopy=True)
            self.assembler.update_state(r, theta, self.t)
            self.assembler.assemble_input()
            self.assembler.assemble_output()
            self.assembler.cache.clear()

        # update past solution cache
        for n, u_n in enumerate(self.PDE.u_old_arr[:-1]):
            u_n.assign(self.PDE.u_old_arr[n + 1])

        self.PDE.u_old_arr[-1].assign(u)

        return


