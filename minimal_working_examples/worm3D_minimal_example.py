from fenics import *
import numpy as np
from worm_rod_engine.parameter.dimensionless_parameter import default_dimensionless_parameter

def grad(f):
    return Dx(f, 0)

#================================================================================================
# Constants
#================================================================================================

e1 = Constant((1.0, 0.0))
e2 = Constant((0.0, 1.0))
e3 = Constant((0.0, 0.0, 1.0))

e = default_dimensionless_parameter.e
alpha = default_dimensionless_parameter.alpha
beta = default_dimensionless_parameter.beta
rho = default_dimensionless_parameter.rho
K_c = default_dimensionless_parameter.K_n
K_n = default_dimensionless_parameter.K_n
K_y = default_dimensionless_parameter.K_y
S = Constant(4.0 / (e ** 2 * alpha) * np.diag([rho, rho, 1]))
S_tilde = Constant(4.0 * beta / (e ** 2 * alpha) * np.diag([rho, rho, 1]))
B = Constant(1.0 / alpha)
B_tilde = Constant(beta / alpha)

#================================================================================================
# Function space
#================================================================================================

N = 200
dt = 0.01

mesh = UnitIntervalMesh(N - 1)
# Spatial element
dx = Measure('dx', domain=mesh)
# Finite elements for 1 dimensional spatial coordinate s
P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
# State variables r and theta are 3 dimensional vector-valued functions of s
P1_3 = MixedElement([P1] * 3)
# Function space for scalar functions of s
V = FunctionSpace(mesh, P1)
# Function space for 3 component vector-valued functions of s
V3 = FunctionSpace(mesh, P1_3)
# Trial function space for 6 component vector-valued function composed of r and theta
W = FunctionSpace(mesh, MixedElement([P1_3, P1_3]))

#================================================================================================
# Weak form
#================================================================================================

u = TrialFunction(W)
phi = TestFunction(W)
r, theta = split(u)
phi_r, phi_theta = split(phi)
u_old = Function(W)
r_old, theta_old = split(u_old)

r_t = (r - r_old) / dt
theta_t = (theta - theta_old) / dt

# We aprroximate head function as solution from previous time/iteration step to linearize the weak form
u_h = Function(W)
r_h, theta_h = split(u_h)

# Head expressions are linear in unkowns
alpha_h, beta_h, gamma_h = theta_h[0], theta_h[1], theta_h[2]

# Rotation matrix
R_x = as_matrix(
    [
        [1, 0, 0],
        [0, cos(gamma_h), -sin(gamma_h)],
        [0, sin(gamma_h), cos(gamma_h)]
    ]
)
R_y = as_matrix(
    [
        [cos(beta_h), 0, sin(beta_h)],
        [0, 1, 0],
        [-sin(beta_h), 0, cos(beta_h)]
    ]
)

R_z = as_matrix(
    [
        [cos(alpha_h), -sin(alpha_h), 0],
        [sin(alpha_h), cos(alpha_h), 0],
        [0, 0, 1]
    ]
)

Q_h =  R_z * R_y * R_x

A_h = as_matrix(
    [
        [0, sin(alpha_h), -cos(alpha_h) * cos(beta_h)],
        [0, -cos(alpha_h), -sin(alpha_h) * cos(beta_h)],
        [-1, 0, sin(beta_h)],
    ]
)

alpha_t, beta_t = theta_t[0], theta_t[1]

A_t = as_matrix(
    [
        [
            0,
            cos(alpha_h) * alpha_t,
            sin(alpha_h) * cos(beta_h) * alpha_t - cos(alpha_h) * sin(beta_h) * beta_t,
        ],
        [
            0,
            sin(alpha_h) * alpha_t,
            - cos(alpha_h) * cos(beta_h) * alpha_t + sin(alpha_h) * sin(beta_h) * beta_t,
        ],

        [0, 0, cos(beta_h) * beta_t],
    ]
)

x_h, y_h, z_h = r_h[0], r_h[1], r_h[2]

T_h = as_matrix(
    [
        [0, -grad(z_h), grad(y_h)],
        [grad(z_h), 0, -grad(x_h)],
        [-grad(y_h), grad(x_h), 0]
    ]
)

# Angular velocity vector
w = A_h * theta_t
# Strain vector
eps = Q_h * grad(r) - e3
# Generalized curvature vector
k = A_h * grad(theta)
# Strain rate vector
eps_t = Q_h * grad(r_t) - cross(w, Q_h * grad(r_h))
# Curvature rate vector
k_t = A_h * grad(theta_t) + A_t * grad(theta_h)
# Internal force
N = Q_h.T * (S * eps + S_tilde * eps_t)

k0 = Expression(('A*sin(q*x[0]-2*pi*t)', '0', '0'), degree=1, A=2*np.pi, q=2*np.pi, t=0.0)

M_and_L_M = Q_h.T * (B * k + B_tilde * k_t)

d3 = Q_h.T * e3
d3d3_h = outer(d3, d3)
f = -(d3d3_h + K_c * (Identity(3) - d3d3_h)) * r_t

# External fluid drag torque
l = (Q_h, w)
e3e3 = outer(e3, e3)
l = -Q_h.T * K_n * (e3e3 + K_y * (Identity(3) - e3e3)) * w

# linear balance
eq1 = dot(f, phi_r) * dx - dot(N, grad(phi_r)) * dx
# Angular balance
eq2 = (
        dot(l, phi_theta) * dx
        + dot(T_h * N, phi_theta) * dx
        - dot(M_and_L_M, grad(phi_theta)) * dx
)

equation = eq1 + eq2

F_op, L = lhs(equation), rhs(equation)

#================================================================================================
# Assign initial values
#================================================================================================

r0_expr = Expression(('0', '0', 'x[0]'), degree=1)
theta0_expr = Expression(('0', '0', '0'), degree=1)
r0, theta0 = Function(V3), Function(V3)
r0.assign(r0_expr)
theta0.assign(theta0_expr)
fa = FunctionAssigner(W, [V3, V3])
#for u_old_n in u_old_arr:
#    fa.assign(u_old_n, [r0, theta0])
fa.assign(u_old, [r0, theta0])

#================================================================================================
# Solve
#================================================================================================
T_sim = 5
for t in np.arange(dt, T_sim+0.1*dt, dt):
    print(f't={t}')
    #k0.t = t
    u_h.assign(u_old)
    u = Function(W)
    solve(F_op == L, u)
    u_old.assign(u)










