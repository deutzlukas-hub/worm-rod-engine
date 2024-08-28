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
    output_param = output_parameter_parser.parse_args(['--k', '--k0'])
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

    # Plot outputs
    gs = plt.GridSpec(5, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[4])

    for r_arr in FS.r[::50, :, :]:
        ax0.plot(r_arr[1, :], r_arr[2, :])

    r_com = FS.r.mean(axis = 2)
    r_mp = FS.r[:, :, FS.r.shape[2]//2]

    U = np.linalg.norm(np.gradient(r_com, worm.dt, axis=0), axis=1)
    ax2.plot(FS.t, U)

    ax1.plot(r_com[:, 1], r_com[:, 2])
    ax1.plot(r_mp[:, 1], r_mp[:, 2])
    plot_scalar_field(ax3, FS.k0[:, 0, :], extent=[0.0, T_sim, 0.0, 1.0])
    plot_scalar_field(ax4, FS.k[:, 0, :], extent=[0.0, T_sim, 0.0, 1.0])

    plt.show()

if __name__ == '__main__':

    simulate_planar_undulation()
