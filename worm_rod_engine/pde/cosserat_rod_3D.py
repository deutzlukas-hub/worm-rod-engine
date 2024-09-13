# From built-in
from typing import Optional, Union, TYPE_CHECKING
from types import SimpleNamespace

import numpy as np
# From third-party
from fenics import *
# From worm-rod-engine
from worm_rod_engine.pde.util import grad, finite_backwards_difference, tag_function_space
from worm_rod_engine.pde.cosserat_rod import PDE_Cosserat
from worm_rod_engine.parameter.output_parameter import FUNCTION_KEYS

if False:
    from fenics import dx

if TYPE_CHECKING:
    from worm_rod_engine.worm import Worm

# Lab frame
e1 = Constant((1, 0, 0))
e2 = Constant((0, 1, 0))
e3 = Constant((0, 0, 1))

#================================================================================================
# Constants
#================================================================================================

class CosseratRod3D(PDE_Cosserat):

    #================================================================================================
    # Default initial state
    #================================================================================================
    def r0_default(self):
        return Expression(('0', '0', 'x[0]'), degree=self.worm.numerical_param.fed)

    def theta0_default(self):
        return Expression(('0', '0', '0'), degree=self.worm.numerical_param.fed)

    #================================================================================================
    # Constants
    #================================================================================================

    def _init_S(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Shear/stretch rigidity matrix
        """
        S = Constant(4.0 / (self.e ** 2 * self.alpha) * np.diag([self.rho, self.rho, 1]))
        if phi is not None: S = phi ** 2 * S

        return S
    def _init_S_tilde(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Shear/stretch damping coefficient matrix
        """
        S_tilde = Constant(4.0 * self.beta / (self.e ** 2 * self.alpha) * np.diag([self.rho, self.rho, 1]))
        if phi is not None: S_tilde = phi ** 2 * S_tilde

        return S_tilde

    def _init_B(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Bending rigidity matrix
        """
        B = Constant(1.0 / self.alpha * np.diag([1, 1, self.rho]))
        if phi is not None: B = phi ** 4 * B

        return B

    def _init_B_tilde(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Bending damping coefficient matrix
        """

        B_tilde = Constant(self.beta / self.alpha * np.diag([1, 1, self.rho]))
        if phi is not None: B_tilde = phi ** 4 * B_tilde

        return B_tilde

    def _init_function_space(self):
        '''
        Initialise finite element function spaces
        '''
        self.mesh = UnitIntervalMesh(self.worm.numerical_param.N - 1)
        # Spatial element
        self.dx = Measure('dx', domain=self.mesh)
        # Finite elements for 1 dimensional spatial coordinate s
        P1 = FiniteElement(self.worm.numerical_param.fet, self.mesh.ufl_cell(), self.worm.numerical_param.fed)
        # State variables r and theta are 3 dimensional vector-valued functions of s
        P1_3 = MixedElement([P1] * 3)
        # Function space for scalar functions of s
        self.V = FunctionSpace(self.mesh, P1)
        # Function space for 3 component vector-valued functions of s
        self.V3 = FunctionSpace(self.mesh, P1_3)
        # Trial function space for 6 component vector-valued function composed of r and theta
        self.W = FunctionSpace(self.mesh, MixedElement([P1_3, P1_3]))

        # Define function spaces of all relevant quantities
        self.function_spaces = {}
        self.function_spaces['r'] = self.function_spaces['theta'] = self.function_spaces['eps0'] = self.function_spaces['k0'] = self.V3

    def _init_form(self):
        """
        Weak form of PDE
        """
        u = TrialFunction(self.W)
        phi = TestFunction(self.W)

        r, theta = split(u)
        phi_r, phi_theta = split(phi)

        # Past states
        self.u_old_arr = [Function(self.W) for _ in np.arange(self.worm.numerical_param.fdo)]

        r_old_arr = [split(u)[0] for u in self.u_old_arr]
        theta_old_arr = [split(u)[1] for u in self.u_old_arr]

        # First time derivatives
        r_t = finite_backwards_difference(1, self.worm.numerical_param.fdo, r, r_old_arr, self.worm.numerical_param.dt)
        theta_t = finite_backwards_difference(1, self.worm.numerical_param.fdo, theta, theta_old_arr, self.worm.numerical_param.dt)

        # Head functions are approximated by previous time/iteration step to linearize the equations of motion
        self.u_h = Function(self.W)
        r_h, theta_h = split(self.u_h)

        # Head expressions are linear in unkowns
        Q_h = self.Q(theta_h)
        A_h = self.A(theta_h)
        A_h_t = self.A_t(theta_h, theta_t)
        T_h = self.T(r_h)

        # Angular velocity vector
        w = self.w(A_h, theta_t)
        # Strain vector
        eps = self.eps(Q_h, r)
        # Generalized curvature vector
        k = self.k(A_h, theta)
        # Strain rate vector
        eps_t = self.eps_t(Q_h, r_h, r_t, w)
        # Curvature rate vector
        k_t = self.k_t(A_h, A_h_t, theta_h, theta_t)
        # Internal force
        N = self.N(Q_h, eps, eps_t)

        # Internal force and muscle force
        if self.eps0 is None:
            N_and_F_M = N
        else:
            N_and_F_M = self.N_and_F_M(Q_h, eps, eps_t)

        # Internal torque and muscle torque
        if self.k0 is None:
            M_and_L_M = self.M(Q_h, k, k_t)
        else:
            M_and_L_M = self.M_and_L_M(Q_h, k, k_t)

        # External fluid drag torque
        l = self.l(Q_h, w)
        # External fluid drag force
        f = self.f(Q_h, r_t)
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

#================================================================================================
# Matrices
#================================================================================================

    @staticmethod
    def Q(theta: Function):
        '''
        Matrix Q rotates lab frame to the body frame
        '''

        alpha, beta, gamma = theta[0], theta[1], theta[2]

        R_x = as_matrix(
            [
                [1, 0, 0],
                [0, cos(gamma), -sin(gamma)],
                [0, sin(gamma), cos(gamma)]
            ]
        )
        R_y = as_matrix(
            [
                [cos(beta), 0, sin(beta)],
                [0, 1, 0],
                [-sin(beta), 0, cos(beta)]
            ]
        )

        R_z = as_matrix(
            [
                [cos(alpha), -sin(alpha), 0],
                [sin(alpha), cos(alpha), 0],
                [0, 0, 1]
            ]
        )
        return R_z * R_y * R_x


    @staticmethod
    def A(theta: Function):
        """
        Curvature-Angular velocity matrix A relates curvature k and angular velocity w to spatial and time derivative of the
        Euler angle vector theta.
        """
        alpha, beta = theta[0], theta[1]

        A = as_matrix(
            [
                [0, sin(alpha), -cos(alpha) * cos(beta)],
                [0, -cos(alpha), -sin(alpha) * cos(beta)],
                [-1, 0, sin(beta)],
            ]
        )
        return A

    @staticmethod
    def A_t(theta: Function, theta_t: Function):
        """
        Time derivative of Curvature-Angular velocity matrix A
        """

        alpha, beta, _ = split(theta)
        alpha_t, beta_t = theta_t[0], theta_t[1]

        A_t = as_matrix(
            [
                [
                    0,
                    cos(alpha) * alpha_t,
                    sin(alpha) * cos(beta) * alpha_t - cos(alpha) * sin(beta) * beta_t,
                    ],
                [
                    0,
                    sin(alpha) * alpha_t,
                    - cos(alpha) * cos(beta) * alpha_t + sin(alpha) * sin(beta) * beta_t ,
                    ],

                [0, 0, cos(beta) * beta_t],
            ]
        )

        return A_t

    @staticmethod
    def T(r: Function):
        '''
        Matrix representation of centreline tangent cross product
        '''

        x, y, z = r[0], r[1], r[2]

        T = as_matrix(
            [[0, -grad(z), grad(y)],
             [grad(z), 0, -grad(x)],
             [-grad(y), grad(x), 0]]
        )


        return T
    #================================================================================================
    # Velocities
    #================================================================================================

    @tag_function_space('V3')
    def r_t(self, r):
        '''
        Centreline velocity
        '''
        r_old_arr = [split(u)[0] for u in self.u_old_arr]
        r_t = finite_backwards_difference(1, self.worm.numerical_param.fdo, r, r_old_arr)
        return r_t


    @staticmethod
    @tag_function_space('V3')
    def w(A, theta_t):
        '''
        Angular velocity
        '''
        return A * theta_t


    #================================================================================================
    # Shape variables and rates
    #================================================================================================

    @staticmethod
    @tag_function_space('V3')
    def eps(Q, r):
        '''
        Strain vector
        '''
        return Q * grad(r) - e3

    @staticmethod
    @tag_function_space('V3')
    def k(A, theta):
        '''
        Generalized curvature vector
        '''
        return A * grad(theta)

    @staticmethod
    @tag_function_space('V3')
    def eps_t(Q, r, r_t, w):
        '''
        Strain rate vector
        '''
        return Q * grad(r_t) - cross(w, Q * grad(r))

    @staticmethod
    @tag_function_space('V3')
    def k_t(A, A_t, theta, theta_t):
        '''
        Time derivative of curvature vector
        '''
        return A * grad(theta_t) + A_t * grad(theta)

    #================================================================================================
    # Forces and torques
    #================================================================================================
    @tag_function_space('V3')
    def f(self, Q, r_t):
        '''
        Fluid drag force line density
        '''
        d3 = Q.T * e3
        d3d3 = outer(d3, d3)
        return -(d3d3 + self.K_c * (Identity(3) - d3d3)) * r_t

    @tag_function_space('V3')
    def l(self, Q, w):
        '''
        Fluid drag force line density
        '''
        e3e3 = outer(e3, e3)
        return -Q.T * self.K_n * (e3e3 + self.K_y * (Identity(3) - e3e3)) * w

    @tag_function_space('V3')
    def N(self, Q, eps, eps_t):
        '''
        Internal force resultant
        '''
        return Q.T * (self.S * eps + self.S_tilde * eps_t)

    @tag_function_space('V3')
    def F_M(self, r_t):
        """
        Muscle force
        """
        return - self.S * self.eps0 * r_t

    @tag_function_space('V3')
    def N_and_F_M(self, Q, eps, eps_t):
        '''
        Internal force resultant and muscle force
        '''
        return Q.T * (self.S * (eps - self.eps0) + self.S_tilde * eps_t)

    @tag_function_space('V3')
    def M(self, Q, k, k_t):
        """
        Internal torque resultant
        """
        return Q.T * (self.B * k + self.B_tilde * k_t)

    @tag_function_space('V3')
    def L_M(self, w):
        """
        Muscle force
        """
        return - self.B * self.k0 * w

    @tag_function_space('V3')
    def M_and_L_M(self, Q, k, k_t):
        '''
        Internal torque resultant and muscle torque
        '''
        return Q.T * (self.B * (k - self.k0) + self.B_tilde * k_t)

    #================================================================================================
    # Energies and powers
    #================================================================================================

    def V(self, k, eps):
        '''
        Calculate elastic energy
        '''

        return 0.5 * assemble( (dot(k, self.B * k) + dot(eps, self.S * eps)) * dx)

    @staticmethod
    def D_E_dot(f, l, r_t, w):
        '''
        Calculate fluid dissipation rate
        '''
        return assemble( ( dot(f, r_t) + dot(l, w) ) * dx)

    def D_I_dot(self, eps_t, k_t):
        '''
        Calculate internal dissipation rate
        '''

        return -assemble( (dot(eps_t, self.S_tilde * eps_t) + dot(k_t, self.B_tilde * k_t)) * dx)

    def V_dot(self, eps, k, eps_t, k_t):
        '''
        Calculate rate of cfrom dolfin import Measure
hange in potential energy
        '''
        return assemble((dot(k, self.B * k_t) + dot(eps, self.S * eps_t)) * dx)

    @staticmethod
    def W_dot(Q, F_M, L_M, r_t, w):
        '''
        Calculate mechanical muscle power
        '''
        return assemble( ( dot(grad(F_M), Q * r_t) + dot(grad(L_M), w) * dx) )





