from typing import Literal

import marsilea as ma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from marsilea.base import StackBoard
from marsilea.plotter import Box, Boxen, Chunk, Violin

from ._palettes import default_palette


class ViolinPlot:
    """
    Expression plot for multiple genes using Marsilea StackBoard.

    Generates one ClusterBoard per (gene x group), then stacks cell types per gene,
    and stacks those gene blocks vertically (or horizontally if `flip=True`).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : list[str]
        List of gene names to plot.
    groupby : str
        Column name in `adata.obs` to group by (e.g., cell type).
    splitby : str | None, optional
        Column name in `adata.obs` to split by (e.g., condition). If None, no splitting is done.
    use_raw : bool, default False
        Whether to use `adata.raw` for expression data. If False, uses `adata.X`.
    figsize : tuple, default (4, 3)
        Size of each individual plot in inches (width, height).
    colors : list[str] | None, optional
        List of colors for each group. If None, uses default colors from `adata.obs[groupby + "_colors"]` or a default palette.
    flip : bool, default False
        If True, flips the orientation of the plots (horizontal vs vertical).
    include_groups : list[str] | None, optional
        List of groups to include in the plot. If None, includes all groups present in `adata.obs[groupby]`.
    include_splits : list[str] | None, optional
        List of splits to include in the plot. If None, includes all splits present in `adata.obs[splitby]`.
    groupby : str
        Column name in `adata.obs` to group by (e.g., cell type).
    splitby : str | None, optional
        Column name in `adata.obs` to split by (e.g., condition). If None, no splitting is done.
    use_raw : bool, default False
        Whether to use `adata.raw` for expression data. If False, uses `adata.X`.
    figsize : tuple, default (4, 3)
        Size of each individual plot in inches (width, height).
    colors : list[str] | None, optional
        List of colors for each group. If None, uses default colors from `adata.obs[groupby + "_colors"]` or a default palette.
    flip : bool, default False
        If True, flips the orientation of the plots (horizontal vs vertical).
    include_groups : list[str] | None, optional
        List of groups to include in the plot. If None, includes all groups present in `adata.obs[groupby]`.
    include_splits : list[str] | None, optional
        List of splits to include in the plot. If None, includes all splits present in `adata.obs[splitby]`.
    """

    def __init__(
        self,
        adata: AnnData,
        genes: list[str],
        groupby: str,
        splitby: str | None = None,
        use_raw: bool = False,
        figsize: tuple = (4, 3),
        colors: list[str] | None = None,
        flip: bool = False,
        include_groups: list[str] | None = None,
        include_splits: list[str] | None = None,
    ):
        self.adata = adata
        self.genes = genes
        self.groupby = groupby
        self.splitby = splitby
        self.use_raw = use_raw
        self.flip = flip
        self.figsize = figsize[::-1] if flip else figsize
        self.include_groups = include_groups or []
        self.include_splits = include_splits or []

        # With this:
        obs_groups = self.adata.obs[groupby].astype(str).unique().tolist()
        if include_groups:
            self.group_values = [g for g in include_groups if g in obs_groups]
        else:
            if pd.api.types.is_categorical_dtype(self.adata.obs[groupby]):
                self.group_values = self.adata.obs[groupby].cat.categories.tolist()
            else:
                self.group_values = sorted(obs_groups)

        # Resolve group colors
        if colors is not None:
            self.group_colors = dict(zip(self.group_values, colors, strict=False))
        elif groupby + "_colors" in self.adata.uns:
            self.group_colors = {
                g: c
                for g, c in zip(
                    self.adata.obs[groupby].cat.categories, self.adata.uns[groupby + "_colors"], strict=False
                )
                if g in self.group_values
            }
        else:
            default_colors = default_palette(len(self.group_values))
            self.group_colors = dict(zip(self.group_values, default_colors, strict=False))

        # Prepare split values and colors
        if self.splitby:
            obs_splits = self.adata.obs[self.splitby].astype(str).unique().tolist()
            if include_splits:
                self.split_values = [s for s in include_splits if s in obs_splits]
            elif pd.api.types.is_categorical_dtype(self.adata.obs[self.splitby]):
                self.split_values = self.adata.obs[self.splitby].cat.categories.tolist()
            else:
                self.split_values = sorted(obs_splits)
            # Resolve split colors
            if self.splitby + "_colors" in self.adata.uns:
                # Try using color mapping from AnnData if available
                categories = (
                    self.adata.obs[self.splitby].cat.categories
                    if pd.api.types.is_categorical_dtype(self.adata.obs[self.splitby])
                    else self.split_values
                )
                raw_colors = self.adata.uns[self.splitby + "_colors"]
                self.split_colors = {
                    k: c for k, c in zip(categories, raw_colors, strict=False) if k in self.split_values
                }
            else:
                self.split_colors = dict(zip(self.split_values, default_palette(len(self.split_values)), strict=False))
        else:
            self.split_values = ["all"]
            self.split_colors = {"all": "#808080"}  # default gray

    def _get_expression_df(self, gene: str) -> pd.DataFrame:
        if self.use_raw and self.adata.raw is not None:
            expr = self.adata.raw[:, gene].X
        else:
            expr = self.adata[:, gene].X

        expr = expr.toarray().flatten() if hasattr(expr, "toarray") else np.ravel(expr)
        df = pd.DataFrame({gene: expr})
        df["group"] = self.adata.obs[self.groupby].astype(str).values
        df["split"] = self.adata.obs[self.splitby].astype(str).values if self.splitby else "all"

        # Apply filters
        df = df[df["group"].isin(self.group_values)]
        df = df[df["split"].isin(self.split_values)]
        return df

    def _build_plot(
        self,
        plot_type: Literal["violin", "box", "boxen"] = "violin",
        show: bool = True,
    ) -> StackBoard:
        outer_boards = []

        plot_class = {
            "violin": Violin,
            "box": Box,
            "boxen": Boxen,
        }.get(plot_type)
        if plot_class is None:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        orient = "v" if self.flip else "h"
        direction = "horizontal" if self.flip else "vertical"

        for gene in self.genes:
            df = self._get_expression_df(gene)
            inner_boards = []

            for group in self.group_values:
                df_sub = df[df["group"] == group].copy()
                if df_sub.empty:
                    continue  # skip if filtered out

                split_dict = {
                    split: gdf[[gene]] for split, gdf in df_sub.groupby("split") if split in self.split_values
                }
                if not split_dict:
                    continue

                matrix = pd.concat(split_dict.values(), axis=0).T

                plot = plot_class(
                    split_dict,
                    orient=orient,
                    hue_order=self.split_values,
                    label=f"{gene} Expression",
                    legend_kws={"title": self.splitby},
                    legend=True,
                    palette=self.split_colors,
                )

                chunk = Chunk([group], [self.group_colors[group]], padding=10)
                cb = ma.ClusterBoard(
                    matrix.T if self.flip else matrix,
                    width=self.figsize[0],
                    height=self.figsize[1],
                )
                cb.add_layer(plot, legend=True)
                if group == self.group_values[-1]:
                    cb.add_legends()
                if self.flip:
                    cb.add_bottom(chunk)
                else:
                    cb.add_left(chunk)
                inner_boards.append(cb)

            if inner_boards:
                gene_stack = StackBoard(inner_boards, direction=direction, spacing=0.5, keep_legends=True)
                outer_boards.append(gene_stack)

        final_stack = StackBoard(
            outer_boards, direction="horizontal" if not self.flip else "vertical", spacing=0.5, keep_legends=True
        )

        return final_stack


def violinplot(
    adata: AnnData,
    genes: list[str],
    groupby: str,
    splitby: str | None = None,
    plot_type: Literal["violin", "box", "boxen"] = "violin",
    save: str | None = None,
    show: bool = True,
    **kwargs,
) -> StackBoard:
    """
    Create a violin plot for gene expression across groups and splits.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : list[str]
        List of gene names to plot.
    groupby : str
        Column name in `adata.obs` to group by.
    splitby : str | None, optional
        Column name in `adata.obs` to split by.
    plot_type : Literal["violin", "box", "boxen"], default "violin"
        Type of plot to generate.
    save : str | None, optional
        Path to save the figure. If None, does not save.
    show : bool, default True
        Whether to display the plot.
    **kwargs
        Additional arguments passed to ViolinPlot.
    """
    plot = ViolinPlot(adata=adata, genes=genes, groupby=groupby, splitby=splitby, **kwargs)
    board = plot._build_plot(plot_type=plot_type, show=show)
    if show:
        with plt.rc_context(rc={"axes.grid": False, "axes.facecolor": "white", "axes.edgecolor": "black"}):
            board.render()
    if save is not None:
        with plt.rc_context(rc={"axes.grid": False, "axes.facecolor": "white", "axes.edgecolor": "black"}):
            board.save(save, bbox_inches="tight")
