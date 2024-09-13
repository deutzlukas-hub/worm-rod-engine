# From third-party
import numpy as np
from fenics import Expression
import matplotlib.pyplot as plt
# From worm-rod-engine
from worm_rod_engine.parameter.output_parameter import output_parameter_parser
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.worm import Worm
from worm_rod_engine.visualize.plot import plot_scalar_field

def simulate_planar_undulation():

    # Specify outputs of interest
    output_param = output_parameter_parser.parse_args(['--k'])
    numerical_param = numerical_argument_parser.parse_args(['--dt', '1e-2', '--N', '250'])
    worm = Worm(numerical_param=numerical_param, output_param=output_param)

    # Simulation time
    T_sim = 1.0
    # Define inputs
    A0, lam0 = 2* np.pi, 1.0
    q0 = 2 * np.pi / lam0
    k0 = Expression(('A0*sin(q0*x[0]+2*pi*t)', '0', '0'), degree=1, t=0.0, A0=A0, q0=q0)
    # Run simulation
    output = worm.solve(5, k0=k0, progress=True)
    assert output[0], 'Simulation failed'
    FS = output[1]

    # Post-process outputs
    r_com = FS.r.mean(axis = 2) # centroid
    r_mp = FS.r[:, :, FS.r.shape[2]//2] # midpoint
    U = np.linalg.norm(np.gradient(r_com, worm.dt, axis=0), axis=1) # swimming speed
    U_avg = U.mean() # average swimming speed

    # Plot outputs
    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0, 0])
    ax10 = plt.subplot(gs[1, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax11 = plt.subplot(gs[1, 1])

    ax01.plot(r_com[:, 1], r_com[:, 2])
    ax01.plot(r_mp[:, 1], r_mp[:, 2])
    ax11.plot(FS.t, U)
    ax11.plot([FS.t[0], FS.t[-1]], [U_avg, U_avg])

    abs_k_max = np.abs([FS.k0.max(), FS.k.max(), FS.k0.min(), FS.k.min()]).max()
    v_lim = [-abs_k_max, abs_k_max]

    plot_scalar_field(ax00, FS.k0[:, 0, :], v_lim=v_lim, extent=[0.0, T_sim, 0.0, 1.0])
    plot_scalar_field(ax10, FS.k[:, 0, :],  v_lim=v_lim, extent=[0.0, T_sim, 0.0, 1.0])

    plt.show()

def simulate_planar_undulation_2D():

    # Specify outputs of interest
    output_param = output_parameter_parser.parse_args(['--k'])
    numerical_param = numerical_argument_parser.parse_args(['--dt', '1e-2', '--N', '250'])
    worm = Worm(dimension=2, numerical_param=numerical_param, output_param=output_param)

    # Simulation time
    T_sim = 1.0
    # Define inputs
    A0, lam0 = 2* np.pi, 1.0
    q0 = 2 * np.pi / lam0
    k0 = Expression('A0*sin(q0*x[0]+2*pi*t)', degree=1, t=0.0, A0=A0, q0=q0)
    # Run simulation
    output = worm.solve(5, k0=k0, progress=True)
    assert output[0], 'Simulation failed'
    FS = output[1]

    # Post-process outputs
    r_com = FS.r.mean(axis = 2) # centroid
    r_mp = FS.r[:, :, FS.r.shape[2]//2] # midpoint
    U = np.linalg.norm(np.gradient(r_com, worm.dt, axis=0), axis=1) # swimming speed
    U_avg = U.mean() # average swimming speed

    # Plot outputs
    gs = plt.GridSpec(2, 2)
    ax00 = plt.subplot(gs[0, 0])
    ax10 = plt.subplot(gs[1, 0])
    ax01 = plt.subplot(gs[0, 1])
    ax11 = plt.subplot(gs[1, 1])

    ax01.plot(r_com[:, 0], r_com[:, 1])
    ax01.plot(r_mp[:, 0], r_mp[:, 1])
    ax11.plot(FS.t, U)
    ax11.plot([FS.t[0], FS.t[-1]], [U_avg, U_avg])

    abs_k_max = np.abs([FS.k0.max(), FS.k.max(), FS.k0.min(), FS.k.min()]).max()
    v_lim = [-abs_k_max, abs_k_max]

    plot_scalar_field(ax00, FS.k0, v_lim=v_lim, extent=[0.0, T_sim, 0.0, 1.0])
    plot_scalar_field(ax10, FS.k,  v_lim=v_lim, extent=[0.0, T_sim, 0.0, 1.0])

    plt.show()

    return

if __name__ == '__main__':

    #simulate_planar_undulation()
    simulate_planar_undulation_2D()