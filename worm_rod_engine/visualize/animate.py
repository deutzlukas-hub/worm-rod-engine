# from built-in
from typing import Optional
# from third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_2D_body(
    r_arr: np.ndarray,
    phi_arr: Optional[np.ndarray] = None,
    fps: Optional[float] = 100.0,
    centered: bool = True,
    plot_outline: bool = False,
    d2_arr: Optional[np.ndarray] = None,
    R_arr: Optional[np.ndarray] = None,
):

    if plot_outline:
        assert d2_arr is not None
        assert R_arr is not None

    # Set up the figure and axis
    fig, ax = plt.subplots()
    centreline, = ax.plot([], [], '-', lw=2, c='r')  # Placeholder for the centerline

    if centered:
        r_arr -= r_arr.mean(axis=-1)[:, :, None]

    x_min = r_arr[:, 0, :].min()
    x_max = r_arr[:, 0, :].max()
    y_min = r_arr[:, 1, :].min()
    y_max = r_arr[:, 1, :].max()

    if plot_outline:
        dorsal_outline, = ax.plot([], [], '-', lw=2, c='k')
        ventral_outline, = ax.plot([], [], '-', lw=2, c='k')
        X_d_arr = r_arr + R_arr[None, None, :] * d2_arr
        X_v_arr = r_arr - R_arr[None, None :] * d2_arr

        x_d_min, x_d_max = X_d_arr[:, 0, :].min(), X_d_arr[:, 0, :].max()
        x_v_min, x_v_max = X_v_arr[:, 0, :].min(), X_d_arr[:, 0, :].max()
        y_d_min, y_d_max = X_d_arr[:, 1, :].min(), X_d_arr[:, 1, :].max()
        y_v_min, y_v_max = X_v_arr[:, 1, :].min(), X_d_arr[:, 1, :].max()

        x_min, x_max = np.min([x_min, x_d_min, x_v_min]), np.max([x_max, x_d_max, x_v_max])
        y_min, y_max = np.min([y_min, y_d_min, y_v_min]), np.max([y_max, y_d_max, y_v_max])

    margin = 0.05

    x_lim = [x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)]
    y_lim = [y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)]

    # Initialize the plot
    def init():
        ax.set_xlim(x_lim)  # Adjust these limits to fit your data
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$', fontsize=14)
        ax.set_ylabel('$y$', fontsize=14)

        if plot_outline:
            return centreline, dorsal_outline, ventral_outline
        return centreline

    # Update function for each frame
    def update(frame):
        centreline.set_data(r_arr[frame, 0, :],  r_arr[frame, 1, :])

        if plot_outline:
            dorsal_outline.set_data(X_d_arr[frame, 0, :], X_d_arr[frame, 1, :])
            ventral_outline.set_data(X_v_arr[frame, 0, :], X_v_arr[frame, 1, :])
            return centreline, dorsal_outline, ventral_outline
        return centreline

    # Create the animation
    ani = FuncAnimation(fig, update, frames=r_arr.shape[0], init_func=init, blit=True, interval=1000/fps)

    # Display the animation
    plt.show()

