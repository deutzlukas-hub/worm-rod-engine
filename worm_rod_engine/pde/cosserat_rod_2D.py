# From built-in
from typing import Optional, Union, TYPE_CHECKING
from types import SimpleNamespace

import numpy as np
# From third-party
from fenics import *
# From worm-rod-engine
from worm_rod_engine.pde.util import grad, finite_backwards_difference, tag_function_space
from worm_rod_engine.util import v2f
from worm_rod_engine.frame import Frame
from worm_rod_engine.pde.cosserat_rod import PDE_Cosserat

if TYPE_CHECKING:
    from worm_rod_engine.worm import Worm

e1 = Constant((1, 0))
e2 = Constant((0, 1))

# "Cross-product" 2D
X = Constant(np.array([[0.0 , -1.0], [1.0,  +0.0]]))

if False:
    from fenics import dx

class CosseratRod2D(PDE_Cosserat):

    #================================================================================================
    # Default initial configuration
    #================================================================================================

    def r0_default(self):
        return Expression(('x[0]', '0'), degree=self.worm.numerical_param.fed)

    def theta0_default(self):
        return Expression('0', degree=self.worm.numerical_param.fed)

    # ================================================================================================
    # Constants
    # ================================================================================================

    def _init_S(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Shear/stretch rigidity matrix
        """
        S = Constant(4.0 / (self.e ** 2 * self.alpha) * np.diag([self.rho, 1]))
        if phi is not None: S = phi ** 2 * S

        return S

    def _init_S_tilde(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Shear/stretch damping coefficient matrix
        """
        S_tilde = Constant(4.0 * self.beta / (self.e ** 2 * self.alpha) * np.diag([self.rho, 1]))
        if phi is not None: S_tilde = phi ** 2 * S_tilde

        return S_tilde

    def _init_B(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Bending rigidity matrix
        """
        B = Constant(1.0 / self.alpha)
        if phi is not None: B = phi ** 4 * B

        return B

    def _init_B_tilde(self, phi: Optional[Union[Expression, Function]] = None):
        """
        Bending damping coefficient matrix
        """

        B_tilde = Constant(self.beta / self.alpha)
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
        # State variables r and theta are 2 dimensional and vector-valued and scalar functions of s
        P1_2 = MixedElement([P1] * 2)
        # Function space for scalar functions of s
        self.V = FunctionSpace(self.mesh, P1)
        # Function space for 2 component vector-valued functions of s
        self.V2 = FunctionSpace(self.mesh, P1_2)
        # Trial function space for 6 component vector-valued function composed of r and theta
        self.W = FunctionSpace(self.mesh, MixedElement(P1_2, P1))

        # Define function spaces of all relevant quantities
        self.function_spaces = {}
        self.function_spaces['r'], self.function_spaces['theta'] = self.V2, self.V
        self.function_spaces['eps0'], self.function_spaces['k0'] = self.V2, self.V

        return


    def _init_form(self):
        """
        Weak form of PDE
        """
        u = TrialFunction(self.W)
        phi = TestFunction(self.W)

        r, theta = split(u)
        phi_r, phi_theta = split(phi)

        # Head functions are approximated by previous time/iteration step to linearize the equations of motion
        self.u_h = Function(self.W)
        r_h, theta_h = split(self.u_h)

        # Past states
        self.u_old_arr = [Function(self.W) for _ in np.arange(self.worm.numerical_param.fdo)]

        r_old_arr = [split(u)[0] for u in self.u_old_arr]
        theta_old_arr = [split(u)[1] for u in self.u_old_arr]

        # Centreline velocity
        r_t = finite_backwards_difference(1, self.worm.numerical_param.fdo, r, r_old_arr, self.worm.numerical_param.dt)
        # Angular velocity vector
        theta_t = finite_backwards_difference(1, self.worm.numerical_param.fdo, theta, theta_old_arr, self.worm.numerical_param.dt)

        # Head expressions are linear in unkowns
        Q_h = self.Q(theta_h)

        # Strain vector
        eps = self.eps(Q_h, r)
        # Generalized curvature vector
        k = grad(theta)
        # Strain rate vector
        eps_t = self.eps_t(Q_h, r_h, r_t, theta_t)
        # Curvature rate vector
        k_t = self.k_t(theta_t)
        # Internal force
        N = self.N_(eps, eps_t)
        # Internal force and muscle force
        if self.eps0 is None:
            N_and_F_M = N
        else:
            N_and_F_M = self.N_and_F_M(eps, eps_t)

        # Internal torque and muscle torque
        if self.k0 is None:
            M_and_L_M = self.M(k, k_t)
        else:
            M_and_L_M = self.M_and_L_M(k, k_t)

        # External fluid drag torque
        l = self.l(theta_t)
        # External fluid drag force
        f = self.f(Q_h, r_t)

        # if self.numerical_param.external_force:
        #     # External fluid drag torque
        #     l = self.l(theta_t)
        #     # External fluid drag force
        #     f = self.f(Q_h, r_t)
        # else:
        # l = Function(self.V)
        # f = Function(self.V2)

        # linear balance
        eq1 = dot(f, phi_r) * dx - dot(Q_h * N_and_F_M, grad(phi_r)) * dx
        # Angular balance
        eq2 = (
                l * phi_theta * dx
                + dot(X * grad(r_h), Q_h * N) * phi_theta * dx
                - M_and_L_M * grad(phi_theta) * dx
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
        return as_matrix(
            [
                [cos(theta), -sin(theta)],
                [sin(theta), +cos(theta)]
            ]
        )

    #================================================================================================
    # Frame
    #================================================================================================

    @staticmethod
    @tag_function_space('V2')
    def d1(Q):
        """
        Dorsal-ventral body-frame vector
        """
        return Q*e1

    @staticmethod
    @tag_function_space('V2')
    def d2(Q):
        """
        Normal body-frame vector
        """
        return Q*e2

    #================================================================================================
    # Velocities
    #================================================================================================
    @tag_function_space('V2')
    def r_t(self, r):
        '''
        Centreline velocity
        '''
        r_old_arr = [split(u)[0] for u in self.u_old_arr]
        r_t = finite_backwards_difference(1, self.worm.numerical_param.fdo, r, r_old_arr, self.worm.numerical_param.dt)
        return r_t

    @tag_function_space('V')
    def theta_t(self, theta):
        '''
        Angular velocity
        '''
        theta_old_arr = [split(u)[1] for u in self.u_old_arr]
        theta_t = finite_backwards_difference(1, self.worm.numerical_param.fdo, theta, theta_old_arr, self.worm.numerical_param.dt)
        return theta_t

    #================================================================================================
    # Shape variables and rates
    #================================================================================================

    @staticmethod
    @tag_function_space('V2')
    def eps(Q, r):
        '''
        Strain vector
        '''
        return Q.T * grad(r) - e1

    @staticmethod
    @tag_function_space('V')
    def k(theta):
        '''
        Generalized curvature vector
        '''
        return grad(theta)

    @staticmethod
    @tag_function_space('V2')
    def eps_t(Q, r, r_t, theta_t):
        '''
        Strain rate vector
        '''
        return Q.T * grad(r_t) - theta_t * X * Q.T * grad(r)

    @staticmethod
    @tag_function_space('V')
    def k_t(theta_t):
        '''
        Time derivative of curvature vector
        '''
        return grad(theta_t)

    #================================================================================================
    # Forces and torques
    #================================================================================================
    @tag_function_space('V2')
    def f(self, Q, r_t):
        '''
        Fluid drag force line density
        '''
        d1 = Q * e1
        d1d1 = outer(d1, d1)
        return -(d1d1 + self.K_c * (Identity(2) - d1d1)) * r_t

    @tag_function_space('V')
    def l(self, theta_t):
        '''
        Fluid drag force line density
        '''
        return -self.K_n * theta_t

    @tag_function_space('V2')
    def N_(self, eps, eps_t):
        '''
        Internal force resultant
        '''
        return self.S * eps + self.S_tilde * eps_t

    @tag_function_space('V2')
    def F_M(self, r_t):
        """
        Muscle force
        """
        if self.eps0 is None:
            F_M = Function(self.V2)
            F_M.assign(Constant((0.0, 0.0)))
        else:
            F_M = project(- self.S * self.eps0, self.V2)
        return F_M

    @tag_function_space('V2')
    def N_and_F_M(self, eps, eps_t):
        '''
        Internal force resultant and muscle force
        '''
        return self.S * (eps - self.eps0) + self.S_tilde * eps_t

    @tag_function_space('V')
    def M(self, k, k_t):
        """
        Internal torque resultant
        """
        return self.B * k + self.B_tilde * k_t

    @tag_function_space('V')
    def L_M(self):
        """
        Muscle force
        """
        if self.k0 is None:
            L_M = Function(self.V)
            L_M.assign(Constant(0.0))
        else:
            L_M = project(- self.B * self.k0, self.V)
        return L_M

    @tag_function_space('V')
    def M_and_L_M(self, k, k_t):
        '''
        Internal torque resultant and muscle torque
        '''
        return self.B * (k - self.k0) + self.B_tilde * k_t

    #================================================================================================
    # Energies and powers
    #================================================================================================

    def V(self, k, eps):
        '''
        Calculate elastic energy
        '''
        return 0.5 * assemble( (dot(k, self.B * k) + dot(eps, self.S * eps)) * dx)

    @staticmethod
    def DE_dot(f, l, r_t, theta_t):
        '''
        Calculate fluid dissipation rate
        '''
        return assemble( ( dot(f, r_t) + dot(l, theta_t) ) * dx)

    def DI_dot(self, eps_t, k_t):
        '''
        Calculate internal dissipation rate
        '''
        return -assemble( (dot(eps_t, self.S_tilde * eps_t) + dot(k_t, self.B_tilde * k_t)) * dx)

    def V_dot(self, eps, k, eps_t, k_t):
        '''
        Calculate rate of cfrom dolfin import Measure hange in potential energy
        '''
        return assemble((dot(k, self.B * k_t) + dot(eps, self.S * eps_t)) * dx)

    @staticmethod
    def W_dot(Q, F_M, L_M, r_t, theta_t):
        '''
        Calculate mechanical muscle power
        '''
        return assemble( (dot(grad(F_M), Q.T * r_t) + dot(grad(L_M), theta_t)) * dx)

