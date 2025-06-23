from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import matplotlib.patches as patches
import numpy as np
from matplotlib import colormaps, rcParams
from matplotlib.colors import to_rgba

from .. import _logger as logg
from .._settings import settings
from .._utils import _empty
from ._anndata import _plot_dendrogram
from ._baseplot_class import BasePlot
from ._utils import (
    _dk,
    check_colornorm,
    cluster_and_reorder_expression_matrix,
    fix_kwds,
    make_grid_spec,
    savefig_or_show,
)

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

    def __init__(
        self,
        adata: AnnData,
        var_names: _VarNames | Mapping[str, _VarNames],
        groupby: str | Sequence[str],
        *,
        use_raw: bool | None = None,
        log: bool = False,
        show_row_names: bool = True,
        show_col_names: bool = True,
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
        cluster_genes: bool = True,
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
            if cluster_genes:
                values_df = cluster_and_reorder_expression_matrix(values_df)

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

        # Width reserved for annotation tiles (in data units)
        self.annotation_width = self._group_annotation_df.shape[1] if hasattr(self, "_group_annotation_df") else 0
        self.dendrogram = dendrogram

        self.show_row_names = show_row_names
        self.show_col_names = show_col_names

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

    def add_annotations(
        self,
        df: pd.DataFrame | None = None,
        palette: dict[str, dict[str, ColorLike] | Colormap] | None = None,
    ) -> None:
        r"""Add annotations to the matrix plot."""
        if df is not None:
            self._group_annotation_df = df
        if palette is not None:
            self._group_annotation_palette = palette

    @property
    def has_annotations(self):
        return hasattr(self, "_group_annotation_df") and self._group_annotation_df is not None

    @property
    def annotation_width(self):
        return self._group_annotation_df.shape[1] if self.has_annotations else 0

    @annotation_width.setter
    def annotation_width(self, value):
        self._annotation_width = value

    def add_dendrogram(
        self,
        *,
        show: bool | None = True,
        dendrogram_key: str | None = None,
        size: float | None = 0.8,
    ) -> Self:
        r"""Show dendrogram based on the hierarchical clustering between the `groupby` categories.

        Categories are reordered to match the dendrogram order.

        The dendrogram information is computed using :func:`scanpy.tl.dendrogram`.
        If `sc.tl.dendrogram` has not been called previously the function is called
        with default parameters.

        The dendrogram is by default shown on the right side of the plot or on top
        if the axes are swapped.

        `var_names` are reordered to produce a more pleasing output if:
            * The data contains `var_groups`
            * the `var_groups` match the categories.

        The previous conditions happen by default when using Plot
        to show the results from :func:`~scanpy.tl.rank_genes_groups` (aka gene markers), by
        calling `scanpy.tl.rank_genes_groups_(plot_name)`.


        Parameters
        ----------
        show
            Boolean to turn on (True) or off (False) 'add_dendrogram'
        dendrogram_key
            Needed if `sc.tl.dendrogram` saved the dendrogram using a key different
            than the default name.
        size
            size of the dendrogram. Corresponds to width when dendrogram shown on
            the right of the plot, or height when shown on top. The unit is the same
            as in matplotlib (inches).

        Returns
        -------
        Returns `self` for method chaining.


        Examples
        --------
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc68k_reduced()
        >>> markers = {"T-cell": "CD3D", "B-cell": "CD79A", "myeloid": "CST3"}
        >>> plot = sc.pl._baseplot_class.BasePlot(adata, markers, groupby="bulk_labels").add_dendrogram()
        >>> plot.plot_group_extra  # doctest: +NORMALIZE_WHITESPACE
        {'kind': 'dendrogram',
         'width': 0.8,
         'dendrogram_key': None,
         'dendrogram_ticks': array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])}

        """
        if not show:
            self.plot_group_extra = None
            return self

        if self.groupby is None or len(self.categories) <= 2:
            # dendrogram can only be computed  between groupby categories
            logg.warning("Dendrogram not added. Dendrogram is added only when the number of categories to plot > 2")
            return self

        self.group_extra_size = size

        # to correctly plot the dendrogram the categories need to be ordered
        # according to the dendrogram ordering.
        self._reorder_categories_after_dendrogram(dendrogram_key)

        dendro_ticks = np.arange(len(self.categories)) + 0.5

        self.group_extra_size = size
        self.plot_group_extra = {
            "kind": "dendrogram",
            "width": size,
            "dendrogram_key": dendrogram_key,
            "dendrogram_ticks": dendro_ticks,
        }
        return self

    def make_figure(self) -> None:
        category_height = self.DEFAULT_CATEGORY_HEIGHT
        category_width = self.DEFAULT_CATEGORY_WIDTH

        # Compute mainplot dimensions
        if self.height is None:
            mainplot_height = len(self.categories) * category_height
            mainplot_width = len(self.var_names) * category_width
            if self.are_axes_swapped:
                mainplot_height, mainplot_width = mainplot_width, mainplot_height

            self.height = max(self.min_figure_height, mainplot_height + 1)
            self.width = mainplot_width + self.legends_width
        else:
            mainplot_height = self.height
            mainplot_width = self.width - self.legends_width

        return_ax_dict = {}

        # Heights: [spacer, var_groups, heatmap]
        var_groups_height = category_height if self.var_groups else 0
        if self.var_groups and not self.are_axes_swapped:
            var_groups_height = category_height / 2

        spacer_height = self.height - var_groups_height - mainplot_height
        height_ratios = [spacer_height, var_groups_height, mainplot_height]

        # == Outer grid for annotation + mid block (main+dendrogram) + legend ==
        ncols_outer = 3 if self.has_annotations else 2
        width_ratios_outer = []
        col_indices_outer = {}

        col = 0
        if self.has_annotations:
            width_ratios_outer.append(self.annotation_width)
            col_indices_outer["annotation"] = col
            col += 1

        mid_width = (
            self.width - self.annotation_width - self.legends_width
            if self.has_annotations
            else self.width - self.legends_width
        )
        width_ratios_outer.append(mid_width)
        col_indices_outer["mid"] = col
        col += 1

        width_ratios_outer.append(self.legends_width)
        col_indices_outer["legend"] = col

        self.fig, outer_gs = make_grid_spec(
            self.ax or (self.width, self.height),
            nrows=3,
            ncols=ncols_outer,
            height_ratios=height_ratios,
            width_ratios=width_ratios_outer,
            hspace=0.0,
            wspace=0.1,
        )

        # Annotation axis in outer grid
        annotation_ax = None
        if "annotation" in col_indices_outer:
            annotation_ax = self.fig.add_subplot(outer_gs[2, col_indices_outer["annotation"]])
            return_ax_dict["annotation_ax"] = annotation_ax

        # == Mid grid inside outer grid for main plot + dendrogram ==
        ncols_mid = 2 if self.dendrogram else 1
        width_ratios_mid = [mainplot_width]
        col_indices_mid = {"main": 0}
        col = 1

        if self.dendrogram:
            width_ratios_mid.append(0.5)
            col_indices_mid["dendrogram"] = col

        mid_gs = outer_gs[2, col_indices_outer["mid"]].subgridspec(
            nrows=1,
            ncols=ncols_mid,
            width_ratios=width_ratios_mid,
            wspace=0.0,
        )

        # Main axis
        main_ax = self.fig.add_subplot(mid_gs[0, col_indices_mid["main"]])
        return_ax_dict["mainplot_ax"] = main_ax

        # Dendrogram axis
        if "dendrogram" in col_indices_mid:
            group_extra_ax = self.fig.add_subplot(mid_gs[0, col_indices_mid["dendrogram"]], sharey=main_ax)
            if self.plot_group_extra["kind"] == "dendrogram":
                _plot_dendrogram(
                    group_extra_ax,
                    self.adata,
                    self.groupby,
                    dendrogram_key=self.plot_group_extra["dendrogram_key"],
                    ticks=self.plot_group_extra["dendrogram_ticks"],
                    orientation="right" if not self.are_axes_swapped else "top",
                )
            elif self.plot_group_extra["kind"] == "group_totals":
                self._plot_totals(group_extra_ax, orientation="right")
            return_ax_dict["group_extra_ax"] = group_extra_ax

        # Legend axis
        legend_ax = self.fig.add_subplot(outer_gs[2, col_indices_outer["legend"]])
        self._plot_legend(legend_ax, return_ax_dict, None)

        # Variable group brackets (top middle panel)
        if self.var_groups:
            brackets_ax = self.fig.add_subplot(outer_gs[1, col_indices_outer["mid"]])
            orientation = "right" if self.are_axes_swapped else "top"
            _plot_var_groups_brackets(
                brackets_ax,
                var_groups=self.var_groups,
                rotation=self.var_group_rotation,
                left_adjustment=0.2,
                right_adjustment=0.7,
                orientation=orientation,
                wide=True,
            )
            return_ax_dict["gene_group_ax"] = brackets_ax

        # Clean all axes
        for ax in self.fig.axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Main matrix plot (heatmap + optional annotation)
        normalize = self._mainplot(main_ax, annotation_ax=annotation_ax)
        main_ax.set_zorder(100)

        if self.fig_title and self.fig_title.strip():
            title_ax = self.fig.add_subplot(outer_gs[0, col_indices_outer["mid"]])
            title_ax.axis("off")
            title_ax.set_title(self.fig_title)

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
            edgecolor="none",
            linewidth=0.0,
            norm=normalize,
        )
        main_ax.pcolor(_color_df, **kwds)

        y_labels = _color_df.index
        x_labels = _color_df.columns

        y_ticks = np.arange(len(y_labels)) + 0.5
        x_ticks = np.arange(len(x_labels)) + 0.5

        # Only set y-ticks on annotation_ax
        main_ax.set_xticks(x_ticks)
        main_ax.set_xticklabels(x_labels, rotation=90, ha="center", minor=False)

        if self.show_row_names and annotation_ax is None:
            main_ax.set_yticks(y_ticks)
            main_ax.set_yticklabels(y_labels)
        else:
            main_ax.set_yticks([])
            main_ax.set_yticklabels([])

        if self.show_col_names:
            main_ax.set_xticks(x_ticks)
            main_ax.set_xticklabels(x_labels)
        else:
            main_ax.set_xticks([])
            main_ax.set_xticklabels([])

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
                col_palette_raw = (
                    self._group_annotation_palette.get(col) if hasattr(self, "_group_annotation_palette") else None
                )
                # Get observed values in order
                cats = annot_df[col].astype("category").cat.categories

                if isinstance(col_palette_raw, dict):
                    col_palette = {cat: to_rgba(col_palette_raw.get(cat, "grey")) for cat in cats}
                elif callable(col_palette_raw):  # likely a colormap
                    cmap = col_palette_raw
                    col_palette = {cat: to_rgba(cmap(i / max(len(cats) - 1, 1))) for i, cat in enumerate(cats)}
                else:
                    # Default fallback: tab10
                    col_palette = {cat: to_rgba(colormaps.get_cmap("tab10")(i)) for i, cat in enumerate(cats)}

                for y, value in enumerate(annot_df[col]):
                    annotation_ax.add_patch(
                        patches.Rectangle(
                            (i, len(annot_df) - y - 1),
                            1,
                            1,
                            facecolor=col_palette.get(value, "grey"),
                            linewidth=1.0,
                            edgecolor=None,
                        )
                    )

            annotation_ax.set_xticks(np.arange(n_annots) + 0.5)
            annotation_ax.set_xticklabels(annot_df.columns, rotation=90, fontsize="x-small")

            if self.show_row_names:
                annotation_ax.set_yticks(y_ticks)
                annotation_ax.set_yticklabels(y_labels)
            else:
                annotation_ax.set_yticks([])
                annotation_ax.set_yticklabels([])

            annotation_ax.tick_params(axis="both", labelsize="small")
            annotation_ax.set_xlim(0, n_annots)
            annotation_ax.set_ylim(len(y_labels), 0)
            annotation_ax.grid(visible=False)
            for axis in ["top", "bottom", "left", "right"]:
                annotation_ax.spines[axis].set_linewidth(1.5)
        return normalize


def matrixplot(
    adata: AnnData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str],
    *,
    use_raw: bool | None = None,
    log: bool = False,
    show_row_names: bool = True,
    show_col_names: bool = True,
    num_categories: int = 7,
    categories_order: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    dendrogram: bool | str = False,
    title: str | None = None,
    cmap: Colormap | str | None = MatrixPlot.DEFAULT_COLORMAP,
    annotation_palette: dict[str, dict[str, ColorLike] | Colormap] | None = None,
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

    This function is a convenient wrapper for :class:`~dx.pl.MatrixPlot`, allowing visualization of grouped expression values as a matrix heatmap. For more customization, use :class:`~dx.pl.MatrixPlot` directly.

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
        If `return_fig` is True, returns a :class:`~dx.pl.MatrixPlot` object.
        If `show` is False, returns a dictionary of matplotlib axes.
        Otherwise, displays the plot and returns None.

    See Also
    --------
    :class:`~dx.pl.MatrixPlot`
        The MatrixPlot class for advanced customization.

    Examples
    --------
    Basic usage:

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
        sc.pl.matrixplot(adata, markers, groupby='bulk_labels', dendrogram=True)
    """
    mp = MatrixPlot(
        adata,
        var_names,
        groupby=groupby,
        use_raw=use_raw,
        log=log,
        show_row_names=show_row_names,
        show_col_names=show_col_names,
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

    if dendrogram:
        mp.add_dendrogram(dendrogram_key=_dk(dendrogram))
    if swap_axes:
        mp.swap_axes()
    if annotation_palette is not None:
        mp.add_annotations(palette=annotation_palette)
    mp = mp.style(cmap=cmap).legend(title=colorbar_title)
    if return_fig:
        return mp
    mp.make_figure()
    savefig_or_show(MatrixPlot.DEFAULT_SAVE_PREFIX, show=show, save=save)
    show = settings.autoshow if show is None else show
    if show:
        return None
    return mp.get_axes()
