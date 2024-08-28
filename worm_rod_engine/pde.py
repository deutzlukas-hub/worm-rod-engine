# From built-in
import math
from typing import Optional, Union, List, TYPE_CHECKING
from types import SimpleNamespace
# From third-party
import numpy as np
from fenics import *

if TYPE_CHECKING:
    from worm_rod_engine.worm import Worm

if False:
    from fenics import dx

def grad(f):
    return Dx(f, 0)

# Lab frame
e1 = Constant((1, 0, 0))
e2 = Constant((0, 1, 0))
e3 = Constant((0, 0, 1))

#================================================================================================
# finite differences to approximate first derivatives
#================================================================================================

def finite_backwards_difference(n: int, k: int, u: Function, u_old_arr: List[Function], dt: float):
    """
    Approximate nth derivative by finite backwards difference of order k of given fenics function z
    """
    # Weighting coefficients and quintile idx
    c_arr, s_arr = finite_difference_coefficients(n, k)

    u_t = 0
    # Add terms to finite backwards difference from most past to most recent time point
    for s, c in zip(s_arr, c_arr):
        if s == 0:
            u_t += c * u
        else:
            u_t += c * u_old_arr[s]

    u_t = u_t / dt**n

    return u_t


def finite_difference_coefficients(n: int, k: int):
    '''
    Calculates weighting coefficients for finite backwards difference of order k for nth derivative
    '''
    # Number points for required for nth derivative of kth order
    N = n + k

    # Point indexes [-k, ..., -1, 0]
    # 0 = current, -1 = previous time point, ...
    s_arr = np.arange(- N + 1, 1)
    A = np.vander(s_arr, increasing=True).T
    b = np.zeros(N)
    b[n] = math.factorial(n)

    # Weighting coefficients of quintiles
    c_arr = np.linalg.solve(A, b)
    # Fenics can't handle numpy arrays
    c_arr = c_arr.tolist()

    return c_arr, s_arr

#================================================================================================
# Rigidity and damping coefficient matrix
#================================================================================================

def S(e: float, alpha: float, rho: float, phi: Optional[Union[Expression, Function]] = None):
    """
    Shear/stretch rigidity matrix
    """
    S = Constant(4.0 / (e ** 2 * alpha) * np.diag([rho, rho, 1]))
    if phi is not None: S = phi ** 2 * S

    return S

def S_tilde(e, alpha, beta, rho, phi: Optional[Union[Expression, Function]] = None):
    """
    Shear/stretch damping coefficient matrix
    """
    S_tilde = Constant(4.0 * beta / (e ** 2 * alpha) * np.diag([rho, rho, 1]))
    if phi is not None: S_tilde = phi**2 * S_tilde

    return S_tilde

def B(alpha: float, rho, phi: Optional[Union[Expression ,Function]] = None):
    """
    Bending rigidity matrix
    """
    B = Constant(1.0 / alpha * np.diag([1, 1, rho]))
    if phi is not None: B = phi**4 * B

    return B

def B_tilde(alpha: float, beta: float, rho: float, phi: Optional[Union[Expression ,Function]] = None):
    """
    Bending damping coefficient matrix
    """

    B_tilde = Constant(beta / alpha * np.diag([1, 1, rho]))
    if phi is not None: B_tilde = phi**4 * B_tilde

    return B_tilde

class PDE():

    def __init__(self, worm: 'Worm'):

        dp = worm.dimensionless_param

        # Constants
        self.S = S(dp.e, dp.alpha, dp.rho)
        self.S_tilde = S_tilde(dp.e, dp.alpha, dp.beta, dp.rho)
        self.B = B(dp.alpha, dp.rho)
        self.B_tilde = B_tilde(dp.alpha, dp.beta, dp.rho)
        self.K_c = Constant(dp.K_c)
        self.K_y = Constant(dp.K_y)
        self.K_n = Constant(dp.K_n)
        # Inputs
        self.k0 = worm.k0
        self.eps0 = worm.k0

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

        # Cross product matrix

        return as_matrix(
            [[0, -grad(z), grad(y)],
             [grad(z), 0, -grad(x)],
             [-grad(y), grad(x), 0]]
        )

    #================================================================================================
    # Velocities
    #================================================================================================

    @staticmethod
    def w(A, theta_t):
        '''
        Angular velocity
        '''

        return A * theta_t


    #================================================================================================
    # Shape variables and rates
    #================================================================================================

    @staticmethod
    def eps(Q, r):
        '''
        Strain vector
        '''

        return Q * grad(r) - e3

    @staticmethod
    def k(A, theta):
        '''
        Generalized curvature vector
        '''
        return A * grad(theta)

    @staticmethod
    def eps_t(Q, r, r_t, w):
        '''
        Strain rate vector
        '''
        return Q * grad(r_t) - cross(w, Q * grad(r))

    @staticmethod
    def k_t(A, A_t, theta, theta_t):
        '''
        Time derivative of curvature vector
        '''
        return A * grad(theta_t) + A_t * grad(theta)

    #================================================================================================
    # Forces and torques
    #================================================================================================

    def f(self, Q, r_t):
        '''
        Fluid drag force line density
        '''
        d3 = Q.T * e3
        d3d3 = outer(d3, d3)
        return -(d3d3 + self.K_c * (Identity(3) - d3d3)) * r_t

    def l(self, Q, w):
        '''
        Fluid drag force line density
        '''
        e3e3 = outer(e3, e3)
        return -Q.T * self.K_n * (e3e3 + self.K_y * (Identity(3) - e3e3)) * w

    def N(self, Q, eps, eps_t):
        '''
        Internal force resultant
        '''
        return Q.T * (self.S * eps + self.S_tilde * eps_t)

    def F_M(self, r_t):
        """
        Muscle force
        """
        return - self.S * self.eps0 * r_t

    def N_and_F_M(self, Q, eps, eps_t):
        '''
        Internal force resultant and muscle force
        '''
        return Q.T * (self.S * (eps - self.eps0) + self.S_tilde * eps_t)

    def M(self, Q, k, k_t):
        """
        Internal torque resultant
        """
        return Q.T * (self.B * k + self.B_tilde * k_t)

    def L_M(self, w):
        """
        Muscle force
        """
        return - self.B * self.k0 * w

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

