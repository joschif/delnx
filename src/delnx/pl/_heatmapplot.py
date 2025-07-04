from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from delnx.pl._baseplot import BasePlot


@dataclass
class HeatmapPlot(BasePlot):
    """HeatmapPlot visualizes grouped single-cell expression data as a heatmap."""

    pass


def heatmap(
    adata: Any,
    markers: Sequence[str],
    groupby: str | list[str],
    save: str | None = None,
    **kwargs,
):
    """
    Create a dot plot showing mean expression and fraction of cells expressing markers per group.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    markers : sequence of str
        Marker genes/features to plot.
    groupby: Key(s) in adata.obs to group by.
    save : str, optional
        If provided, saves the plot to the specified file path.
    **kwargs
        Additional arguments passed to DotPlot.

    Returns
    -------
        None
    """
    plot = HeatmapPlot(adata=adata, markers=markers, groupby_keys=groupby, **kwargs)
    if save:
        plot.save(save, bbox_inches="tight")
    else:
        plot.show()
