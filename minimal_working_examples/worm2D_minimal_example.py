from fenics import *
from dolfin import *
import numpy as np
from worm_rod_engine.parameter.dimensionless_parameter import default_dimensionless_parameter

def grad(f):
    return Dx(f, 0)

#================================================================================================
# Constants
#================================================================================================
e1 = Constant((1.0, 0.0))
e2 = Constant((0.0, 1.0))

X = Constant(np.array([
    [0.0 , -1.0],
    [1.0,  +0.0]
]))

e = default_dimensionless_parameter.e
alpha = default_dimensionless_parameter.alpha
beta = default_dimensionless_parameter.beta
rho = default_dimensionless_parameter.rho
K_c = default_dimensionless_parameter.K_n
K_n = default_dimensionless_parameter.K_n

S = Constant(4.0 / (e ** 2 * alpha) * np.diag([rho, 1]))
S_tilde = Constant(4.0 * beta / (e ** 2 * alpha) * np.diag([rho, 1]))
B = Constant(1.0 / alpha)
B_tilde = Constant(beta / alpha)

#================================================================================================
# Function spaces
#================================================================================================
N = 250
dt = 0.01

mesh = UnitIntervalMesh(N - 1)
# Spatial element
dx = Measure('dx', domain=mesh)
# Finite elements for 1 dimensional spatial coordinate s
P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
# State variables r and theta are 2 dimensional and vector-valued and scalar functions of s
P1_2 = MixedElement([P1] * 2)
# Function space for scalar functions of s
V = FunctionSpace(mesh, P1)
# Function space for 2 component vector-valued functions of s
V2 = FunctionSpace(mesh, P1_2)
# Trial function space for 6 component vector-valued function composed of r and theta
W = FunctionSpace(mesh, MixedElement([P1_2, P1]))


#================================================================================================
# Weak form
#================================================================================================

u = TrialFunction(W)
phi = TestFunction(W)

r, theta = split(u)
phi_r, phi_theta = split(phi)

#Input
#k0 = Expression('A*sin(q*x[0]-2*pi*t)', degree=1, A=2*np.pi, q=2*np.pi, t=0.0)
k0 = Constant(np.pi)

# Past states
# u_old_arr = [Function(W) for _ in np.arange(2)]
# r_old_arr = [split(u)[0] for u in u_old_arr]
# theta_old_arr = [split(u)[1] for u in u_old_arr]
u_old = Function(W)
r_old, theta_old = split(u_old)

# Centreline velocity
#r_t = finite_backwards_difference(1, 2, r, r_old_arr, dt)
# Angular velocity vector
#w = finite_backwards_difference(1, 2, theta, theta_old_arr, dt)
r_t = (r - r_old) / dt
w = (theta - theta_old) / dt

# Head functions are approximated by previous time/iteration step to linearize the equations of motion
u_h = Function(W)
r_h, theta_h = split(u_h)

Q_h = as_matrix(
    [
        [cos(theta_h), -sin(theta_h)],
        [sin(theta_h), +cos(theta_h)]
    ]
)

# Strain vector
eps = Q_h.T * grad(r) - e1
# Generalized curvature vector
k = grad(theta)
# Strain rate vector
eps_t = Q_h.T * grad(r_t) - w * X * Q_h.T * grad(r_h)
# Curvature rate vector
k_t = grad(w)
# Internal force
N = S * eps + S_tilde * eps_t
# Internal torque and muscle torque
M_and_L_M = B*(k-k0) + B_tilde * k_t

# External fluid drag torque
l = -K_n * w
# External fluid drag force
d1_h = Q_h * e1
d1d1_h = outer(d1_h, d1_h)
f = - (d1d1_h + K_c * (Identity(2) - d1d1_h)) * r_t
# linear balance
eq1 = dot(f, phi_r) * dx - dot(Q_h * N, grad(phi_r)) * dx
# Angular balance
eq2 = (
        l * phi_theta * dx
        + dot(X * grad(r_h), Q_h * N) * phi_theta * dx
        - M_and_L_M * grad(phi_theta) * dx
)

equation = eq1 + eq2

F_op, L = lhs(equation), rhs(equation)

#================================================================================================
# Assign initial values
#================================================================================================
r0_expr = Expression(('x[0]', '0'), degree=1)
theta0_expr = Expression('0', degree=1)
r0, theta0 = Function(V2), Function(V)
r0.assign(r0_expr)
theta0.assign(theta0_expr)
fa = FunctionAssigner(W, [V2, V])
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


