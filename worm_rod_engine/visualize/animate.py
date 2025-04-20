# from built-in
from typing import Optional, List, Tuple
from collections import defaultdict, deque
# from third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
# form paper
from gait_optimality_py.plotting.figure_layout import f_tilde_cmap

def update_swimmer(
    swimmer,
    r_arr: np.ndarray,
    d2_arr: np.ndarray,
    R_arr: np.ndarray,):

    X_d_arr = r_arr + R_arr[None, :] * d2_arr
    X_v_arr = r_arr - R_arr[None, :] * d2_arr
    # Transpose to get into (N, 2) shape
    X_arr = np.vstack((X_d_arr.T, np.flipud(X_v_arr.T)))

    swimmer.set_xy(X_arr)

def plot_cosserat_rod_shape(
    r_arr: np.ndarray,
    d2_arr: np.ndarray,
    R_arr: np.ndarray,
    ax: Optional[Axes] = None,
    facecolor: str = "grey",
    **kwargs):

    if ax is None:
        ax = plt.subplots()

    X_d_arr = r_arr + R_arr[None, :] * d2_arr
    X_v_arr = r_arr - R_arr[None, :] * d2_arr
    # Transpose to get into (N, 2) shape
    X_arr = np.vstack((X_d_arr.T, np.flipud(X_v_arr.T)))

    polygon = Polygon(X_arr, facecolor=facecolor, edgecolor='black', closed=True, **kwargs)
    ax.add_patch(polygon)

    return polygon

def animate_multiple_swimmers_with_different_frequencies(
    r_arr_list: List[np.ndarray],
    d2_arr_list: List[np.ndarray],
    t_arr_list: List[np.ndarray],
    R_arr: np.ndarray,
    f_tilde_arr: np.ndarray,
):

    T_max = min([t_arr[-1] for t_arr in t_arr_list])
    dt_min = min([t_arr[1] - t_arr[0] for t_arr in t_arr_list])

    t_arr_list_masked = []
    r_arr_list_masked = []
    d2_arr_list_masked = []

    x_max, y_max = -100, -100
    x_min, y_min = +100, +100

    for t_arr, r_arr, d2_arr in zip(t_arr_list, r_arr_list, d2_arr_list):

        mask = t_arr <= T_max
        t_arr = t_arr[mask]
        r_arr = r_arr[mask]
        d2_arr = d2_arr[mask]

        r_max, r_min = r_arr.max(axis=(0, -1)), r_arr.min(axis=(0, -1))
        x_max = max(x_max, r_max[0])
        y_max = max(y_max, r_max[1])
        x_min = min(x_min, r_min[0])
        y_min = min(y_min, r_min[1])

        t_arr_list_masked.append(t_arr)
        r_arr_list_masked.append(r_arr)
        d2_arr_list_masked.append(d2_arr)

    # Step 2: Define common timeline
    dt = dt_min / 10.0
    times = np.arange(0, T_max+0.1*dt, dt)
    frames = len(times)

    update_map = defaultdict(list)

    # Iterate over objects and their event times
    for swimmer_idx, t_arr in enumerate(t_arr_list_masked):
        for t in t_arr:
            # Find the closest time step in t_values
            index = np.abs(times - t).argmin()
            if index < len(times):  # Ensure it does not exceed the range
                update_map[times[index]].append(swimmer_idx)


    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim((x_min, x_max))  # Reset limits (optional)
    ax.set_ylim((y_min, y_max))  # Reset limits (optional)

    # Add colorbat for normalized freqency from 0 to 1
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=f_tilde_cmap, norm=norm)
    sm.set_array([])  # No actual data needed
    cbar = fig.colorbar(sm, cax=ax, orientation='vertical', extend='max')
    f_tilde_cmap.set_over('white')

    color_list = sm.to_rgba(f_tilde_arr)

    # Init swimmers
    swimmers = []
    for i, color in enumerate(color_list):
        swimmer_idx = Polygon(np.array([[0, 0], [1,1]]), facecolor=color, edgecolor='black', closed=True)
        swimmers.append(swimmer_idx)
        ax.add_patch(swimmer_idx)

    swimmer_update_idx = [0 for _ in range(len(swimmers))]

    def update(frame):

        print(frames - frame)

        for swimmer_idx in update_map[times[frame]]:

            t_idx = swimmer_update_idx[swimmer_idx]

            r = r_arr_list_masked[swimmer_idx][t_idx]
            d2 = d2_arr_list_masked[swimmer_idx][t_idx]
            update_swimmer(swimmers[swimmer_idx], r, d2, R_arr)

            swimmer_update_idx[swimmer_idx] += 1


    fps = 0.5 * int(1.0 / dt_min)
    interval = 1000 / fps
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    return ani

def animate_multiple_swimmer_in_one_panel(
    r_arr_list: List[np.ndarray],
    d2_arr_list: List[np.ndarray],
    color_list: List,
    R_arr: np.ndarray,
    fps: Optional[float] = 100.0,
    xlim = None,
    ylim = None,
):

    fig, ax = plt.subplots()
    frames = r_arr_list[0].shape[0]

    def update(t):

        ax.clear()  # Clear the axis to reset the plot

        for r_arr, d2_arr, color in zip(r_arr_list, d2_arr_list, color_list):
            r = r_arr[t, :]
            d2 = d2_arr[t, :]
            plot_cosserat_rod_shape(r, d2, R_arr, ax=ax, facecolor=color, alpha=0.3)

        ax.set_aspect('equal')

        if xlim is not None:
            ax.set_xlim(xlim)  # Reset limits (optional)
        if ylim is not None:
            ax.set_ylim(ylim)  # Reset limits (optional)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

    interval = 1000 / fps
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    return ani


def animate_multiple_swimmer(
    rows,
    cols,
    r_arr_list: List[np.ndarray],
    d2_arr_list: List[np.ndarray],
    fps: float,
    titles: Optional[List[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,):

    fig = plt.figure()
    gs = plt.GridSpec(rows, cols)
    axes = []
    for row in range(rows):
        axes.append([])
        for col in range(cols):
            axes[row].append(plt.subplot(gs[row, col]))

    def update(t):
        for row in rows:
            for col in cols:
                ax = axes[row, col]
                r_arr = r_arr_list[row][col]
                d2_arr = d2_arr_list[row][col]

                r = r_arr[t, :]
                d2 = d2_arr[t, :]
                ax.clear()  # Clear the axis to reset the plot
                plot_cosserat_rod_shape(r, d2, R_arr, ax=ax)
                ax.set_aspect('equal')

                if xlim is not None:
                    ax.set_xlim(xlim)  # Reset limits (optional)
                if ylim is not None:
                    ax.set_ylim(ylim)  # Reset limits (optional)

                ax.set_xlabel('x')
                ax.set_ylabel('y')

def animate_planar_rod(
    r_arr: np.ndarray,
    d2_arr: np.ndarray,
    R_arr: np.ndarray,
    fps: Optional[float] = 100.0,
    xlim = None,
    ylim = None,
):

    fig, ax = plt.subplots()
    frames = r_arr.shape[0]

    def update(t):
        r = r_arr[t, :]
        d2 = d2_arr[t, :]
        ax.clear()  # Clear the axis to reset the plot
        plot_cosserat_rod_shape(r, d2, R_arr, ax=ax)
        ax.set_aspect('equal')

        if xlim is not None:
            ax.set_xlim(xlim)  # Reset limits (optional)
        if ylim is not None:
            ax.set_ylim(ylim)  # Reset limits (optional)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

    interval = 1000 / fps
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    return ani

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

