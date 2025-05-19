# from built-in
from typing import Optional, List, Tuple, Callable
from collections import defaultdict, deque
# from third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def animate_swimmers_in_lab_and_body_frame(
    fig: Figure,
    axes: List[Axes],
    r_arr_list: np.ndarray,
    d2_arr_list: np.ndarray,
    R_arr: np.ndarray,
    color_list: List[str],
    alpha: float = 1.0,
    fps: float = 100,
    play_back_rate: float = 1.0,
    head_markers: Optional[List[str]] = None,
    ax_callback_lab_frame: Callable = None,
    ax_callback_body_frame: Callable = None,):

    ax0, ax1 = axes[0], axes[1]

    N_swimmer = len(r_arr_list)
    frame_count = r_arr_list[0].shape[0]

    # Init swimmers
    swimmers_lab_frame, swimmers_body_frame = [], []

    for i, color in enumerate(color_list):
        poly = Polygon(np.array([[0, 0], [1,1]]), facecolor=color, edgecolor='black', closed=True, alpha=alpha)
        ax0.add_patch(poly)
        swimmers_lab_frame.append(poly)

        poly = Polygon(np.array([[0, 0], [1,1]]), facecolor=color, edgecolor='black', closed=True, alpha=alpha)
        ax1.add_patch(poly)
        swimmers_body_frame.append(poly)

    head_markers_lab_frame, head_markers_body_frame = [], []
    # Add marker
    if head_markers is not None:
        for marker, col in head_markers:
            line, = ax0.plot([], [],
                linestyle='None', marker=marker, markerfacecolor=col, edgecolor='black', markersize=5)
            head_markers_lab_frame.append(line)
            line, = ax0.plot([], [],
                linestyle='None', marker=marker, markerfacecolor=col, edgecolor='black', markersize=5)
            head_markers_body_frame.append(line)

    if ax_callback_lab_frame is not None:
        ax_callback_lab_frame(ax0)

    if ax_callback_body_frame is not None:
        ax_callback_body_frame(ax1)

    def update(frame):
        for idx in range(N_swimmer):
            print(f'{frame_count- frame} more frames to render')

            r_arr = r_arr_list[idx][frame]
            d2_arr = d2_arr_list[idx][frame]

            # Lab-frame
            X_d_arr = r_arr + R_arr[None, :] * d2_arr
            X_v_arr = r_arr - R_arr[None, :] * d2_arr
            # Transpose to get into (N, 2) shape
            X_arr = np.vstack((X_d_arr.T, np.flipud(X_v_arr.T)))
            swimmers_lab_frame[idx].set_xy(X_arr)
            # Body-frame
            r_cm = r_arr.mean(axis=-1)
            r_arr -= r_cm[..., None]
            X_d_arr = r_arr + R_arr[None, :] * d2_arr
            X_v_arr = r_arr - R_arr[None, :] * d2_arr
            X_arr = np.vstack((X_d_arr.T, np.flipud(X_v_arr.T)))
            swimmers_body_frame[idx].set_xy(X_arr)

            # TODO
            # if decorate_swimmer_with_marker is not None:
            #     R = r_arr.mean(axis=-1)
            #     for marker in centroid_markers:
            #         marker.set_data([R[0]], [R[1]])


    interval = 1000 / fps / play_back_rate
    ani = FuncAnimation(fig, update, frames=frame_count, interval=interval)

    return ani

def animate_multiple_swimmer_in_multiple_panels(
    fig,
    axes,
    r_arr_list_list: List[List[np.ndarray]],
    d2_arr_list_list: List[List[np.ndarray]],
    R_arr: np.ndarray,
    color_list: List[str],
    alpha: float = 1.0,
    fps: float = 100,
    play_back_rate: float = 1.0,
    ax_callback: Optional[Callable] = None,):

    frame_count = r_arr_list_list[0][0].shape[0]

    swimmers_list = []

    for i, (ax, r_arr_list) in enumerate(zip(axes, r_arr_list_list)):
        swimmers = []
        for j, color in enumerate(color_list):
            poly = Polygon(np.array([[0, 0], [1, 1]]), facecolor=color, edgecolor='black', closed=True, alpha=alpha)
            swimmers.append(poly)
            ax.add_patch(poly)
        swimmers_list.append(swimmers)

    if ax_callback is not None:
        for ax in axes:
            ax_callback(ax)

    def update(frame):

        print(f'{frame_count - frame} more frames to render')

        for i, (swimmers, r_arr_list, d2_arr_list) in enumerate(zip(swimmers_list, r_arr_list_list, d2_arr_list_list)):
            for j, (swimmer, r_arr, d2_arr) in enumerate(zip(swimmers, r_arr_list, d2_arr_list)):

                r_arr = r_arr_list[j][frame]
                d2_arr = d2_arr_list[j][frame]
                X_d_arr = r_arr + R_arr[None, :] * d2_arr
                X_v_arr = r_arr - R_arr[None, :] * d2_arr
                # Transpose to get into (N, 2) shape
                X_arr = np.vstack((X_d_arr.T, np.flipud(X_v_arr.T)))
                swimmers[j].set_xy(X_arr)

    interval = 1000 / fps / play_back_rate
    ani = FuncAnimation(fig, update, frames=frame_count, interval=interval)

    return ani

def animate_multiple_swimmer_in_one_panel(
    fig: Figure,
    ax: Axes,
    r_arr_list: List[np.ndarray],
    d2_arr_list: List[np.ndarray],
    R_arr: np.ndarray,
    color_list: List[str],
    alpha: float = 1.0,
    fps: float = 100,
    play_back_rate: float = 1.0,
    ax_callback: Optional[Callable] = None,
    ):

    N_swimmer = len(r_arr_list)
    frame_count = r_arr_list[0].shape[0]

    # Init swimmers
    swimmers = []
    centroid_markers = []

    for i, color in enumerate(color_list):
        poly = Polygon(np.array([[0, 0], [1,1]]), facecolor=color, edgecolor='black', closed=True, alpha=alpha)
        swimmers.append(poly)
        ax.add_patch(poly)

        # #TODO: Add dynamic markersize
        # if decorate_swimmer_with_marker is not None:
        #     for marker, col in decorate_swimmer_with_marker:
        #         line, = Line2D([], [],
        #                        linestyle='None', marker=marker, markerfacecolor=col, edgecolor='black', markersize=5)
        #         centroid_markers.append(line)


        # if swimmer_labels is not None:
        #     # Create a proxy line for the legend
        #     proxy_line = Line2D([], [],
        #         linestyle='None', marker='o', color='k', markerfacecolor=color, markersize=10, label=swimmer_labels[i])
        #     ax.add_line(proxy_line)

    if ax_callback is not None:
        ax_callback(ax)

    def update(frame):
        for idx in range(N_swimmer):
            print(f'{frame_count - frame} more frames to render')

            r_arr = r_arr_list[idx][frame]
            d2_arr = d2_arr_list[idx][frame]

            X_d_arr = r_arr + R_arr[None, :] * d2_arr
            X_v_arr = r_arr - R_arr[None, :] * d2_arr
            # Transpose to get into (N, 2) shape
            X_arr = np.vstack((X_d_arr.T, np.flipud(X_v_arr.T)))
            swimmers[idx].set_xy(X_arr)

            # if decorate_swimmer_with_marker is not None:
            #     R = r_arr.mean(axis=-1)
            #     for marker in centroid_markers:
            #         marker.set_data([R[0]], [R[1]])

    interval = 1000 / fps / play_back_rate
    ani = FuncAnimation(fig, update, frames=frame_count, interval=interval)
    return ani

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
    t_star_arr_list: List[np.ndarray],
    R_arr: np.ndarray,
    f_arr: np.ndarray,
    color_list: Optional[List] = None,
    alpha: float = 1.0,
    ax_callback: Optional[Callable] = None,
    play_back_rate: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None):

    #================================================================================================
    # Pre-processing
    #================================================================================================

    N_swimmer = len(r_arr_list)

    # fps most resolve fastest motion
    f_max = max(f_arr)
    # Choose frames per dimensionless time
    fps = 25
    dt_star = 1 / fps
    dt = dt_star / f_max

    # Convert to physical time
    t_arr_list = [t_star_arr / f for  t_star_arr, f in zip(t_star_arr_list, f_arr)]
    T_max = min([t_arr[-1] for t_arr in t_arr_list])

    t_arr_list_masked = []
    r_arr_list_masked = []
    d2_arr_list_masked = []

    # Mask data
    for t_arr, r_arr, d2_arr in zip(t_arr_list, r_arr_list, d2_arr_list):

        mask = t_arr <= T_max
        t_arr = t_arr[mask]
        r_arr = r_arr[mask]
        d2_arr = d2_arr[mask]

        t_arr_list_masked.append(t_arr)
        r_arr_list_masked.append(r_arr)
        d2_arr_list_masked.append(d2_arr)

    #x_max = max([r_arr[:, 0, :].max() for r_arr in r_arr_list_masked])
    x_max = 1.0
    x_min = min([r_arr[:, 0, :].min() for r_arr in r_arr_list_masked])
    y_max = max([r_arr[:, 1, :].max() for r_arr in r_arr_list_masked])
    y_min = min([r_arr[:, 1, :].min() for r_arr in r_arr_list_masked])

    # Define global time stamps
    global_t_arr = np.arange(0, T_max + 0.1*dt, dt)
    frames = len(global_t_arr)

    update_map = [[] for _ in range(frames)]

    # Iterate over global time stamps
    for frame, global_t in enumerate(global_t_arr):
        # For each swimmer, find frame whose time stamp is closest to global time stamp
        for swimmer_idx, t_arr in enumerate(t_arr_list_masked):
            t_idx = np.abs(t_arr - global_t).argmin()
            update_map[frame].append(t_idx)

    #================================================================================================
    # Animiation
    #================================================================================================

    if figsize is None:
        # Determine figsize
        dx = x_max - x_min
        dy = y_max - y_min

        aspect_ratio = dy / dx
        w = 19.2
        h = 3 * aspect_ratio * w
        figsize = (w, h)

    #fig = plt.figure(figsize=figsize, dpi=100)
    fig = plt.figure(figsize=figsize, dpi=100)

    ax = plt.subplot(111)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_aspect('equal')

    if ax_callback is not None:
        ax_callback(ax)

    # plt.show()

    if color_list is None:
        color_list = N_swimmer * ['grey']

    # Init swimmers
    swimmers = []

    for i, color in enumerate(color_list):
        poly = Polygon(np.array([[0, 0], [1,1]]), facecolor=color, edgecolor='black', closed=True, alpha=alpha)
        swimmers.append(poly)
        ax.add_patch(poly)

    t_idx_arr = [0 for i in range(N_swimmer)]

    def update(frame):

        print(f'{frames - frame} more frames to go')
        t_idx_list = update_map[frame]

        for swimmer_idx, t_idx in enumerate(t_idx_list):

            r = r_arr_list_masked[swimmer_idx][t_idx]
            d2 = d2_arr_list_masked[swimmer_idx][t_idx]
            update_swimmer(swimmers[swimmer_idx], r, d2, R_arr)

    interval =  1000 / fps / play_back_rate
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

