# From third-party
import numpy as np
from fenics import Constant, Expression
import matplotlib.pyplot as plt
# From worm-rod-engine
from worm_rod_engine.parameter.output_parameter import output_parameter_parser
from worm_rod_engine.parameter.numerical_parameter import numerical_argument_parser
from worm_rod_engine.worm import Worm
from worm_rod_engine.visualize.plot import plot_scalar_field, plot_scalar_fiedls_grid

def actuation_relaxation():

    # Specify outputs of interest
    output_param = output_parameter_parser.parse_args(['--k0', '--k', '--eps'])
    numerical_param = numerical_argument_parser.parse_args(['--dt', '1e-2', '--N', '750'])
    worm = Worm(numerical_param=numerical_param, output_param=output_param)

    # Simulation time
    T_sim = 2.0
    # Define inputs
    k0 = Constant((np.pi, 0, 0))
    k0 = Expression(('pi', '0', '0'), degree=worm.fed)
    # Run simulation
    output = worm.solve(T_sim, k0=k0, progress=True)
    assert output[0], 'Simulation failed'
    FS = output[1]
    plot_scalar_fiedls_grid(2, 3,
        [FS.k0[:, 0, :], FS.k0[:, 1, :], FS.k0[:, 2, :], FS.k[:, 0, :], FS.k[:, 1, :], FS.k[:, 2, :]],
        extent=[0.0, T_sim, 0.0, 1.0])

    plot_scalar_fiedls_grid(1, 3,
        [FS.eps[:, 0, :], FS.eps[:, 1, :], FS.eps[:, 2, :]],
        extent=[0.0, T_sim, 0.0, 1.0])

    plt.show()

    # # Plot outputs
    # gs = plt.GridSpec(3, 1)
    # ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1])
    # ax2 = plt.subplot(gs[2])
    #
    # r_arr = FS.r[-1, :]
    # ax0.plot(r_arr[1, :], r_arr[2, :])
    # plot_scalar_field(ax1, FS.k0[:, 0, :], extent=[0., T_sim, 0., 1.])
    # plot_scalar_field(ax2, FS.k[:, 0, :], extent=[0., T_sim, 0., 1.])
    #
    # plt.show()



if __name__ == '__main__':

    actuation_relaxation()
