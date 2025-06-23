from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from cycler import Cycler, cycler
from matplotlib import axes, gridspec, rcParams
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .. import _logger as logg
from .._settings import settings

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    # TODO: more
    DensityNorm = Literal["area", "count", "width"]

# These are needed by _wraps_plot_scatter
VBound = str | float | Callable[[Sequence[float]], float]
_FontWeight = Literal["light", "normal", "medium", "semibold", "bold", "heavy", "black"]
_FontSize = Literal["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]
_LegendLoc = Literal[
    "none",
    "right margin",
    "on data",
    "on data export",
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]
ColorLike = str | tuple[float, ...]


class _AxesSubplot(Axes, axes.SubplotBase):
    """Intersection between Axes and SubplotBase: Has methods of both."""


# -------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------


def savefig(writekey, dpi=None, ext=None):
    """Save current figure to file.

    The `filename` is generated as follows:

        filename = settings.figdir / (writekey + settings.plot_suffix + '.' + settings.file_format_figs)
    """
    if dpi is None:
        # we need this as in notebooks, the internal figures are also influenced by 'savefig.dpi' this...
        if not isinstance(rcParams["savefig.dpi"], str) and rcParams["savefig.dpi"] < 150:
            if settings._low_resolution_warning:
                logg.warning(
                    "You are using a low resolution (dpi<150) for saving figures.\n"
                    "Consider running `set_figure_params(dpi_save=...)`, which will "
                    "adjust `matplotlib.rcParams['savefig.dpi']`"
                )
                settings._low_resolution_warning = False
        else:
            dpi = rcParams["savefig.dpi"]
    settings.figdir.mkdir(parents=True, exist_ok=True)
    if ext is None:
        ext = settings.file_format_figs
    filename = settings.figdir / f"{writekey}{settings.plot_suffix}.{ext}"
    # output the following msg at warning level; it's really important for the user
    logg.warning(f"saving figure to file {filename}")
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")


def savefig_or_show(
    writekey: str,
    show: bool | None = None,
    dpi: int | None = None,
    ext: str | None = None,
    save: bool | str | None = None,
):
    if isinstance(save, str):
        # check whether `save` contains a figure extension
        if ext is None:
            for try_ext in [".svg", ".pdf", ".png"]:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    save = save.replace(try_ext, "")
                    break
        # append it
        writekey += save
        save = True
    save = settings.autosave if save is None else save
    show = settings.autoshow if show is None else show
    if save:
        savefig(writekey, dpi=dpi, ext=ext)
    if show:
        plt.show()
    if save:
        plt.close()  # clear figure


def default_palette(
    palette: str | Sequence[str] | Cycler | None = None,
) -> str | Cycler:
    if palette is None:
        return rcParams["axes.prop_cycle"]
    elif not isinstance(palette, str | Cycler):
        return cycler(color=palette)
    else:
        return palette


def make_grid_spec(
    ax_or_figsize: tuple[int, int] | _AxesSubplot,
    *,
    nrows: int,
    ncols: int,
    wspace: float | None = None,
    hspace: float | None = None,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
) -> tuple[Figure, gridspec.GridSpecBase]:
    kw = dict(
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    if isinstance(ax_or_figsize, tuple):
        fig = plt.figure(figsize=ax_or_figsize)
        return fig, gridspec.GridSpec(nrows, ncols, **kw)
    else:
        ax = ax_or_figsize
        ax.axis("off")
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)


def fix_kwds(kwds_dict, **kwargs):
    """Merge the parameters into a single consolidated dictionary.

    Given a dictionary of plot parameters (`kwds_dict`) and a dict of `kwds`,
    this function prevents argument duplication errors.

    If `kwds_dict` an kwargs have the same key, only the value in `kwds_dict` is kept.

    Parameters
    ----------
    kwds_dict
        kwds dictionary
    kwargs

    Returns
    -------
    `kwds_dict` merged with `kwargs`

    Examples
    --------
    >>> def _example(**kwds):
    ...     return fix_kwds(kwds, key1="value1", key2="value2")
    >>> _example(key1="value10", key3="value3")
    {'key1': 'value10', 'key2': 'value2', 'key3': 'value3'}

    """
    kwargs.update(kwds_dict)

    return kwargs


def check_colornorm(vmin=None, vmax=None, vcenter=None, norm=None):
    from matplotlib.colors import Normalize

    try:
        from matplotlib.colors import TwoSlopeNorm as DivNorm
    except ImportError:
        # matplotlib<3.2
        from matplotlib.colors import DivergingNorm as DivNorm

    if norm is not None:
        if (vmin is not None) or (vmax is not None) or (vcenter is not None):
            msg = "Passing both norm and vmin/vmax/vcenter is not allowed."
            raise ValueError(msg)
    elif vcenter is not None:
        norm = DivNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    return norm


def _dk(dendrogram: bool | str | None) -> str | None:
    """Convert the `dendrogram` parameter to a `dendrogram_key` parameter."""
    return None if isinstance(dendrogram, bool) else dendrogram
