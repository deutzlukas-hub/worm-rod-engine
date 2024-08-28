# From built-in
from typing import Tuple, List, Optional
# From third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_scalar_field(
        ax: Axes,
        field: np.ndarray,
        v_lim: Tuple[float, float] = None,
        extent: Tuple[float, float, float, float] = None,
        eps: float = 1e-3,
        cmap: str = None,
        **cbar_kwargs,
):
    """
    Plots a colormap representation of the 2D scalar field M onto s and t.
    """
    if v_lim is not None:
        v_min, v_max = v_lim[0], v_lim[1]
    else:
        v_min, v_max = field.min(), field.max()

    if np.abs(v_min) < eps:
        np.sign(v_min) * eps
    if np.abs(v_max) < eps:
        np.sign(v_max) * eps

    m = ax.imshow(field.T, cmap= cmap, aspect="auto", origin="lower", vmin = v_min, vmax = v_max, extent=extent)
    cbar = plt.colorbar(m, ax=ax, **cbar_kwargs)

    if not extent:
        ax.get_yaxis().set_ticks([0.0, 0.5, 1.0])
        ax.get_xaxis().set_ticks([])

    return cbar, m

def plot_scalar_fiedls_grid(
    nrows: int,
    ncols: int,
    fields: List[np.ndarray],
    v_lims: Optional[List[Tuple[float, float]]] = None,
    cmaps: Optional[List[str]] = None,
    **kwargs):

    assert nrows*ncols == len(fields)

    plt.figure()
    gs = plt.GridSpec(nrows, ncols)

    k = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = plt.subplot(gs[i, j])
            cmap = cmaps[k] if cmaps is not None else None
            v_lim = v_lims[k] if v_lims is not None else None
            plot_scalar_field(ax, fields[k], cmap=cmap, v_lim=v_lim, **kwargs)
            k += 1







