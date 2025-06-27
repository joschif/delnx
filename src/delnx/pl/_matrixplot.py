from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import marsilea as ma
import marsilea.plotter as mp
import pandas as pd

from ..pp._utils import group_by_max
from ._baseplot import BasePlot


@dataclass
class MatrixPlot(BasePlot):
    """
    MatrixPlot visualizes group-level mean expression data as a heatmap,
    with support for group annotations and flexible row grouping.

    Parameters
    ----------
        group_metadata : pd.DataFrame
            Metadata for each group, used for annotations.
    """

    group_metadata: pd.DataFrame = field(init=False)

    def _build_data(self) -> pd.DataFrame:
        """
        Computes group-level mean expression and prepares group metadata.

        Returns
        -------
            pd.DataFrame: Mean expression matrix (groups x markers).
        """
        group_col = self.adata.obs["_group"].astype(str)
        df = pd.DataFrame(self.adata[:, self.markers].X.toarray(), index=group_col)
        self.mean_df = df.groupby(df.index).mean()
        self.mean_df.columns = self.markers

        group_meta = (
            self.adata.obs[self.groupby_keys]
            .copy()
            .assign(_group=group_col)
            .drop_duplicates("_group")
            .set_index("_group")
        )
        self.group_metadata = group_meta.loc[list(self.mean_df.index)]

        if self.group_metadata.isnull().any().any():
            missing = self.group_metadata[self.group_metadata.isnull().any(axis=1)].index.tolist()
            raise ValueError(f"Missing group metadata for: {missing}")

        return self.mean_df

    def _resolve_row_grouping(self, index_source: Any | None = None) -> tuple[pd.Categorical | None, list[str] | None]:
        """
        Determines row grouping for the heatmap.

        Parameters
        ----------
            index_source: Any | None
            Optional source for row indices, defaults to mean_df index.

        Returns
        -------
            Tuple of (group labels, group categories) or (None, None).
        """
        if self.row_grouping == "auto":
            return self.mean_df.index, list(self.mean_df.index)

        elif self.row_grouping is None:
            return None, None

        elif isinstance(self.row_grouping, str):
            group = pd.Categorical(self.group_metadata[self.row_grouping])
            return group, group.categories

        elif isinstance(self.row_grouping, list):
            key_df = self.group_metadata[self.row_grouping].astype(str)
            compound = key_df.agg("_".join, axis=1)
            group = pd.Categorical(compound)
            return group, group.categories

        elif isinstance(self.row_grouping, (pd.Series, pd.Categorical)):
            group = (
                self.row_grouping.loc[self.mean_df.index]
                if isinstance(self.row_grouping, pd.Series)
                else self.row_grouping
            )
            group = pd.Categorical(group)
            return group, group.categories

        else:
            raise ValueError("Invalid value for row_grouping in MatrixPlot.")

    def _add_group_colorbar(self, m: ma.Heatmap, key: str):
        """
        Add a colorbar for a specific group key.

        Parameters
        ----------
        m : ma.Heatmap
            The heatmap object to which the colorbar will be added.
        key : str
            The key in `adata.obs` for which to add the colorbar.

        Raises
        ------
        ValueError
            If the key is not found in `adata.obs`.
        """
        values = self.group_metadata[key]
        palette = dict(zip(list(self.adata.obs[key].cat.categories), self.adata.uns.get(f"{key}_colors"), strict=False))
        palette = {e: palette[e] for e in values}
        label = self.group_names[self.groupby_keys.index(key)]
        colorbar = mp.Colors(list(values), palette=palette, label=label)
        m.add_left(colorbar, size=self.groupbar_size, pad=self.groupbar_pad)

    def _build_plot(self):
        """
        Build the plot

        Returns
        -------
        marsilea.SizedHeatmap
            The build matrix plot object.
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

        m = ma.Heatmap(
            data,
            cmap=self.cmap,
            height=self.height,
            width=self.width,
            vmin=self.vmin,
            vmax=self.vmax,
            center=self.center,
            cbar_kws={"title": "Expression\nin group"},
        )

        if self.row_group is not None:
            m.group_rows(self.row_group, order=self.order)

        m = self._add_extras(m)
        return m


def matrixplot(
    adata: Any,
    markers: Sequence[str],
    groupby: str | list[str],
    save: str | None = None,
    **kwargs,
):
    """
    Create a matrix plot showing mean expression of markers per group.

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
    plot = MatrixPlot(adata=adata, markers=markers, groupby_keys=groupby, **kwargs)
    if save:
        plot.save(save, bbox_inches="tight")
    else:
        plot.show()
