import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import marsilea as ma
import numpy as np
import pandas as pd

from delnx.pp._utils import group_by_max

from ._matrixplot import MatrixPlot


@dataclass
class DotPlot(MatrixPlot):
    """
    DotPlot visualizes mean expression and fraction of cells expressing markers per group.

    Inherits from MatrixPlot and uses marsilea's SizedHeatmap for visualization.
    """

    def _build_plot(self):
        """
        Build the plot.

        Returns
        -------
        marsilea.SizedHeatmap
            The build dot plot object.
        """
        data = self._build_data()

        # Resolve row grouping
        self.row_group, self.order = self._resolve_row_grouping(self.mean_df.index.astype(str))

        # Check if dendrogram is specified
        # If yes, we have to precompute the dendrogram
        # since the dendrograms are computed on the fly, we will have to
        # change the ordering of the markers & reextract the matrix
        # needed if column grouping is enabled
        if self.dendrograms and self.column_grouping:
            for pos in self.dendrograms:
                if pos in ["left", "right"]:
                    cb = ma.Heatmap(data)
                    cb.add_dendrogram(pos, add_base=False)

                    deform_order = cb.get_deform()
                    deform_order._run_cluster()

                    row_order = deform_order.row_reorder_index
                    data_reordered = data.iloc[row_order, :]
                    self.markers = group_by_max(data_reordered.T)

                    data = self._build_data()

        # Flatten markers if given as dict
        if isinstance(self.markers, dict):
            flat_markers = list(itertools.chain.from_iterable(self.markers.values()))
        else:
            flat_markers = self.markers

        # Get detection data from raw or specified layer — do not fall back to .X
        if self.adata.raw is not None:
            X_raw = self.adata.raw.to_adata()[:, flat_markers].X.toarray()
        elif self.layer is not None:
            if self.layer not in self.adata.layers:
                raise ValueError(f"Layer '{self.layer}' not found in adata.layers.")
            X_raw = self.adata[:, flat_markers].layers[self.layer]
        else:
            raise ValueError("DotPlot requires either `adata.raw` or a specified `layer`. Neither was provided.")

        group_labels = self.adata.obs["_group"].astype(str)
        df = pd.DataFrame(X_raw > 0, index=group_labels)

        agg_count = df.gt(0).groupby(df.index).sum().loc[self.mean_df.index]
        agg_cell_counts = df.groupby(df.index).size().loc[self.mean_df.index].to_numpy()
        size = agg_count.to_numpy() / agg_cell_counts[:, np.newaxis]

        m = ma.SizedHeatmap(
            size=size,
            color=data,
            sizes=(1, self.scale * 100),
            vmin=self.vmin,
            vmax=self.vmax,
            width=self.width,
            height=self.height,
            center=self.center,
            cmap=self.cmap,
            edgecolor=None,
            linewidth=0,
            color_legend_kws={"title": "Expression\nin group"},
            size_legend_kws={
                "title": "Fraction of cells\nin group (%)",
                "labels": ["20%", "40%", "60%", "80%", "100%"],
                "show_at": [0.2, 0.4, 0.6, 0.8, 1.0],
            },
        )

        m = self._add_extras(m)
        return m


def dotplot(
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
    **kwargs
        Additional arguments passed to DotPlot.
    """
    plot = DotPlot(adata=adata, markers=markers, groupby_keys=groupby, **kwargs)
    if save:
        plot.save(save, bbox_inches="tight")
    else:
        plot.show()
