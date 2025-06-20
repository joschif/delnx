from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import matplotlib.patches as patches
import numpy as np
from matplotlib import colormaps, rcParams
from matplotlib.colors import to_rgba

from .. import _logger as logg
from .._settings import settings
from .._utils import _doc_params, _empty
from ._anndata import _plot_dendrogram
from ._baseplot_class import BasePlot, doc_common_groupby_plot_args
from ._docs import (
    doc_common_plot_args,
    doc_show_save_ax,
    doc_vboundnorm,
)
from ._utils import _dk, check_colornorm, fix_kwds, make_grid_spec, savefig_or_show

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal, Self

    import pandas as pd
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize

    from .._utils import Empty
    from ._baseplot_class import _VarNames
    from ._utils import ColorLike, _AxesSubplot

    _VarNames = str | Sequence[str]


@_doc_params(common_plot_args=doc_common_plot_args)
class MatrixPlot(BasePlot):
    """Allows the visualization of values using a color map.

    Parameters
    ----------
    {common_plot_args}
    title
        Title for the figure.
    expression_cutoff
        Expression cutoff that is used for binarizing the gene expression and
        determining the fraction of cells expressing given genes. A gene is
        expressed only if the expression value is greater than this threshold.
    mean_only_expressed
        If True, gene expression is averaged only over the cells
        expressing the given genes.
    standard_scale
        Whether or not to standardize that dimension between 0 and 1,
        meaning for each variable or group,
        subtract the minimum and divide each by its maximum.
    values_df
        Optionally, a dataframe with the values to plot can be given. The
        index should be the grouby categories and the columns the genes names.

    kwds
        Are passed to :func:`matplotlib.pyplot.scatter`.

    See Also
    --------
    :func:`~scanpy.pl.matrixplot`: Simpler way to call MatrixPlot but with less options.
    :func:`~scanpy.pl.rank_genes_groups_matrixplot`: to plot marker genes identified
        using the :func:`~scanpy.tl.rank_genes_groups` function.

    Examples
    --------
    Simple visualization of the average expression of a few genes grouped by
    the category 'bulk_labels'.

    .. plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
        sc.pl.MatrixPlot(adata, markers, groupby='bulk_labels').show()

    Same visualization but passing var_names as dict, which adds a grouping of
    the genes on top of the image:

    .. plot::
        :context: close-figs

        markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
        sc.pl.MatrixPlot(adata, markers, groupby='bulk_labels').show()

    """

    DEFAULT_SAVE_PREFIX = "matrixplot_"
    DEFAULT_COLOR_LEGEND_TITLE = "Mean expression\nin group"

    # default style parameters
    DEFAULT_COLORMAP = rcParams["image.cmap"]
    DEFAULT_EDGE_COLOR = "gray"
    DEFAULT_EDGE_LW = 0.1

    def __init__(
        self,
        adata: AnnData,
        var_names: _VarNames | Mapping[str, _VarNames],
        groupby: str | Sequence[str],
        *,
        use_raw: bool | None = None,
        log: bool = False,
        num_categories: int = 7,
        categories_order: Sequence[str] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        gene_symbols: str | None = None,
        var_group_positions: Sequence[tuple[int, int]] | None = None,
        var_group_labels: Sequence[str] | None = None,
        var_group_rotation: float | None = None,
        layer: str | None = None,
        standard_scale: Literal["var", "group"] | None = None,
        ax: _AxesSubplot | None = None,
        values_df: pd.DataFrame | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        vcenter: float | None = None,
        norm: Normalize | None = None,
        dendrogram: bool | str = False,
        **kwds,
    ):
        BasePlot.__init__(
            self,
            adata,
            var_names,
            groupby,
            use_raw=use_raw,
            log=log,
            num_categories=num_categories,
            categories_order=categories_order,
            title=title,
            figsize=figsize,
            gene_symbols=gene_symbols,
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=var_group_rotation,
            layer=layer,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            norm=norm,
            **kwds,
        )

        # In case values_df is not provided, we compute it
        if values_df is None:
            # compute mean value
            values_df = (
                self.obs_tidy.groupby(level=0, observed=True)
                .mean()
                .loc[self.categories_order if self.categories_order is not None else self.categories]
            )

            if standard_scale == "group":
                values_df = values_df.sub(values_df.min(1), axis=0)
                values_df = values_df.div(values_df.max(1), axis=0).fillna(0)
            elif standard_scale == "var":
                values_df -= values_df.min(0)
                values_df = (values_df / values_df.max(0)).fillna(0)
            elif standard_scale is None:
                pass
            else:
                logg.warning("Unknown type for standard_scale, ignored")

        self.values_df = values_df

        self.cmap = self.DEFAULT_COLORMAP
        self.edge_color = self.DEFAULT_EDGE_COLOR
        self.edge_lw = self.DEFAULT_EDGE_LW

        # Width reserved for annotation tiles (in data units)
        self.annotation_width = self._group_annotation_df.shape[1] if hasattr(self, "_group_annotation_df") else 0
        self.dendrogram = dendrogram

        if self.dendrogram:
            self.plot_group_extra = {
                "kind": "dendrogram",
                "dendrogram_key": self.dendrogram if isinstance(dendrogram, str) else None,
            }
        else:
            self.plot_group_extra = None

    def style(
        self,
        cmap: Colormap | str | None | Empty = _empty,
        edge_color: ColorLike | None | Empty = _empty,
        edge_lw: float | None | Empty = _empty,
    ) -> Self:
        r"""Modify plot visual parameters.

        Parameters
        ----------
        cmap
            Matplotlib color map, specified by name or directly.
            If ``None``, use :obj:`matplotlib.rcParams`\ ``["image.cmap"]``
        edge_color
            Edge color between the squares of matrix plot.
            If ``None``, use :obj:`matplotlib.rcParams`\ ``["patch.edgecolor"]``
        edge_lw
            Edge line width.
            If ``None``, use :obj:`matplotlib.rcParams`\ ``["lines.linewidth"]``

        Returns
        -------
        :class:`~scanpy.pl.MatrixPlot`

        Examples
        --------

        .. plot::
            :context: close-figs

            import scanpy as sc

            adata = sc.datasets.pbmc68k_reduced()
            markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']

        Change color map and turn off edges:


        .. plot::
            :context: close-figs

            (
                sc.pl.MatrixPlot(adata, markers, groupby='bulk_labels')
                .style(cmap='Blues', edge_color='none')
                .show()
            )

        """
        super().style(cmap=cmap)

        if edge_color is not _empty:
            self.edge_color = edge_color
        if edge_lw is not _empty:
            self.edge_lw = edge_lw

        return self

    def add_annotations(self, df: pd.DataFrame) -> Self:
        self._group_annotation_df = df
        return self

    @property
    def has_annotations(self):
        return hasattr(self, "_group_annotation_df") and self._group_annotation_df is not None

    @property
    def annotation_width(self):
        return self._group_annotation_df.shape[1] if self.has_annotations else 0

    @annotation_width.setter
    def annotation_width(self, value):
        self._annotation_width = value

    def make_figure(self) -> None:
        category_height = self.DEFAULT_CATEGORY_HEIGHT
        category_width = self.DEFAULT_CATEGORY_WIDTH

        # Compute dimensions
        if self.height is None:
            mainplot_height = len(self.categories) * category_height
            mainplot_width = len(self.var_names) * category_width
            if self.are_axes_swapped:
                mainplot_height, mainplot_width = mainplot_width, mainplot_height

            height = mainplot_height + 1  # space for labels
            self.height = max(self.min_figure_height, height)
            self.width = mainplot_width + self.legends_width
        else:
            self.min_figure_height = self.height
            mainplot_height = self.height
            mainplot_width = self.width - self.legends_width

        return_ax_dict = {}

        # Heights: [spacer, var_groups, heatmap]
        var_groups_height = category_height if self.var_groups else 0
        if self.var_groups and not self.are_axes_swapped:
            var_groups_height = category_height / 2

        spacer_height = self.height - var_groups_height - mainplot_height
        height_ratios = [spacer_height, var_groups_height, mainplot_height]

        # Widths: build column structure conditionally
        width_ratios = []
        col_indices = {}  # map logical names to actual grid indices
        col = 0

        if self.has_annotations:
            width_ratios.append(self.annotation_width)
            col_indices["annotation"] = col
            col += 1

        width_ratios.append(mainplot_width)
        col_indices["main"] = col
        col += 1

        if self.dendrogram:
            dendrogram_width = 0.5
            width_ratios.append(dendrogram_width)
            col_indices["dendrogram"] = col
            col += 1

        width_ratios.append(self.legends_width)
        col_indices["legend"] = col
        ncols = len(width_ratios)

        # Build unified grid
        self.fig, gs = make_grid_spec(
            self.ax or (self.width, self.height),
            nrows=3,
            ncols=ncols,
            hspace=0.0,
            wspace=0.1,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        )

        # Optional annotation axis
        if "annotation" in col_indices:
            annot_ax = self.fig.add_subplot(gs[2, col_indices["annotation"]])
            annot_ax.set_xticks([])
            annot_ax.set_yticks([])
            return_ax_dict["annotation_ax"] = annot_ax

        # Main plot axis
        main_ax = self.fig.add_subplot(gs[2, col_indices["main"]])
        return_ax_dict["mainplot_ax"] = main_ax

        # Optional dendrogram axis
        if "dendrogram" in col_indices and getattr(self, "plot_group_extra", None):
            ax = self.fig.add_subplot(gs[2, col_indices["dendrogram"]], sharey=main_ax)
            if self.plot_group_extra["kind"] == "dendrogram":
                _plot_dendrogram(
                    ax,
                    self.adata,
                    self.groupby,
                    dendrogram_key=self.plot_group_extra["dendrogram_key"],
                    orientation="right",
                )
                return_ax_dict["dendrogram_ax"] = ax
            elif self.plot_group_extra["kind"] == "group_totals":
                self._plot_totals(ax, orientation="right")
                return_ax_dict["group_totals_ax"] = ax

        # Optional legend axis
        if self.legends_width > 0:
            legend_ax = self.fig.add_subplot(gs[2, col_indices["legend"]])
            return_ax_dict["legend_ax"] = legend_ax

        # Optional title
        if self.fig_title and self.fig_title.strip():
            title_ax = self.fig.add_subplot(gs[0, col_indices["main"]])
            title_ax.axis("off")
            title_ax.set_title(self.fig_title)

        # Clean all axes
        for ax in self.fig.axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Plot main content
        normalize = self._mainplot(main_ax, annotation_ax=return_ax_dict.get("annotation_ax"))
        main_ax.set_zorder(100)

        # Plot legend (after getting normalization)
        if self.legends_width > 0:
            self._plot_legend(return_ax_dict["legend_ax"], return_ax_dict, normalize)

        self.ax_dict = return_ax_dict

    def _mainplot(self, main_ax: Axes, annotation_ax: Axes | None = None):
        # 1. Setup expression matrix data
        _color_df = self.values_df.copy()
        if self.var_names_idx_order is not None:
            _color_df = _color_df.iloc[:, self.var_names_idx_order]
        if self.categories_order is not None:
            _color_df = _color_df.loc[self.categories_order, :]
        if self.are_axes_swapped:
            _color_df = _color_df.T

        cmap = colormaps.get_cmap(self.kwds.get("cmap", self.cmap))
        if "cmap" in self.kwds:
            del self.kwds["cmap"]

        normalize = check_colornorm(
            self.vboundnorm.vmin,
            self.vboundnorm.vmax,
            self.vboundnorm.vcenter,
            self.vboundnorm.norm,
        )

        for axis in ["top", "bottom", "left", "right"]:
            main_ax.spines[axis].set_linewidth(1.5)

        kwds = fix_kwds(
            self.kwds,
            cmap=cmap,
            edgecolor=self.edge_color,
            linewidth=self.edge_lw,
            norm=normalize,
        )
        main_ax.pcolor(_color_df, **kwds)

        y_labels = _color_df.index
        x_labels = _color_df.columns

        y_ticks = np.arange(len(y_labels)) + 0.5
        x_ticks = np.arange(len(x_labels)) + 0.5

        # Only set y-ticks on annotation_ax
        main_ax.set_yticks([])
        main_ax.set_xticks(x_ticks)
        main_ax.set_xticklabels(x_labels, rotation=90, ha="center", minor=False)

        main_ax.grid(visible=False)
        main_ax.set_ylim(len(y_labels), 0)
        main_ax.set_xlim(0, len(x_labels))

        if annotation_ax is not None and hasattr(self, "_group_annotation_df"):
            annot_df = self._group_annotation_df
            annot_df = annot_df.reindex(y_labels)

            if annot_df.isnull().all().all():
                raise ValueError("Annotation DataFrame has no matching indices with the matrix plot.")

            n_annots = annot_df.shape[1]
            annotation_ax.set_xlim(0, n_annots)
            annotation_ax.set_ylim(0, len(annot_df))

            for i, col in enumerate(annot_df.columns):
                categories = annot_df[col].astype("category").cat.categories
                palette = {cat: to_rgba(colormaps.get_cmap("tab10")(j)) for j, cat in enumerate(categories)}
                for y, value in enumerate(annot_df[col]):
                    annotation_ax.add_patch(
                        patches.Rectangle(
                            (i, len(annot_df) - y - 1),
                            1,
                            1,
                            facecolor=palette.get(value, "grey"),
                            linewidth=1.0,
                        )
                    )

            annotation_ax.set_xticks(np.arange(n_annots) + 0.5)
            annotation_ax.set_xticklabels(annot_df.columns, rotation=90, fontsize="x-small")

            annotation_ax.set_yticks(y_ticks)
            annotation_ax.set_yticklabels(y_labels)

            annotation_ax.tick_params(axis="both", labelsize="small")
            annotation_ax.set_xlim(0, n_annots)
            annotation_ax.set_ylim(len(y_labels), 0)
            annotation_ax.grid(visible=False)
            for axis in ["top", "bottom", "left", "right"]:
                annotation_ax.spines[axis].set_linewidth(1.5)
        return normalize


@_doc_params(
    show_save_ax=doc_show_save_ax,
    common_plot_args=doc_common_plot_args,
    groupby_plots_args=doc_common_groupby_plot_args,
    vminmax=doc_vboundnorm,
)
def matrixplot(
    adata: AnnData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str],
    *,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    categories_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    dendrogram: bool | str = False,
    title: str | None = None,
    cmap: Colormap | str | None = MatrixPlot.DEFAULT_COLORMAP,
    colorbar_title: str | None = MatrixPlot.DEFAULT_COLOR_LEGEND_TITLE,
    gene_symbols: str | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
    var_group_labels: Sequence[str] | None = None,
    var_group_rotation: float | None = None,
    layer: str | None = None,
    standard_scale: Literal["var", "group"] | None = None,
    values_df: pd.DataFrame | None = None,
    swap_axes: bool = False,
    show: bool | None = None,
    save: str | bool | None = None,
    ax: _AxesSubplot | None = None,
    return_fig: bool | None = False,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    norm: Normalize | None = None,
    **kwds,
) -> MatrixPlot | dict[str, Axes] | None:
    """
    Create a heatmap of mean expression values per group for each variable.

    This function is a convenient wrapper for :class:`~scanpy.pl.MatrixPlot`, allowing visualization of grouped expression values as a matrix heatmap. For more customization, use :class:`~scanpy.pl.MatrixPlot` directly.

    Parameters
    ----------
    {common_plot_args}
    {groupby_plots_args}
    {show_save_ax}
    {vminmax}
    cmap
        Colormap to use for the heatmap.
    colorbar_title
        Title for the colorbar legend.
    swap_axes
        If True, swap the axes of the matrix plot.
    values_df
        Optional DataFrame with values to plot (index: groupby categories, columns: gene names).
    kwds
        Additional keyword arguments passed to :func:`matplotlib.pyplot.pcolor`.

    Returns
    -------
    MatrixPlot | dict[str, Axes] | None
        If `return_fig` is True, returns a :class:`~scanpy.pl.MatrixPlot` object.
        If `show` is False, returns a dictionary of matplotlib axes.
        Otherwise, displays the plot and returns None.

    See Also
    --------
    :class:`~scanpy.pl.MatrixPlot`
        The MatrixPlot class for advanced customization.
    :func:`~scanpy.pl.rank_genes_groups_matrixplot`
        Plot marker genes identified by :func:`~scanpy.tl.rank_genes_groups`.

    Examples
    --------
    Basic usage:

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
        sc.pl.matrixplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Using `var_names` as a dictionary:

        markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
        sc.pl.matrixplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Accessing the MatrixPlot object for further customization:

        mp = sc.pl.matrixplot(adata, markers, 'bulk_labels', return_fig=True)
        mp.add_totals().style(edge_color='black').show()

    Getting the axes dictionary:

        axes_dict = mp.get_axes()
    """
    mp = MatrixPlot(
        adata,
        var_names,
        groupby=groupby,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        categories_order=categories_order,
        standard_scale=standard_scale,
        title=title,
        figsize=figsize,
        gene_symbols=gene_symbols,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=var_group_rotation,
        layer=layer,
        values_df=values_df,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        norm=norm,
        dendrogram=dendrogram,
        **kwds,
    )

    if swap_axes:
        mp.swap_axes()
    mp = mp.style(cmap=cmap).legend(title=colorbar_title)
    if return_fig:
        return mp
    mp.make_figure()
    savefig_or_show(MatrixPlot.DEFAULT_SAVE_PREFIX, show=show, save=save)
    show = settings.autoshow if show is None else show
    if show:
        return None
    return mp.get_axes()
