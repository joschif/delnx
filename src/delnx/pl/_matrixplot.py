from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import matplotlib.patches as patches
import numpy as np
from matplotlib import colormaps, gridspec, rcParams
from matplotlib.colors import to_rgba

from .. import _logger as logg
from .._compat import old_positionals
from .._settings import settings
from .._utils import _doc_params, _empty
from ._anndata import (
    _plot_dendrogram,
    _plot_var_groups_brackets,
)
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

    @old_positionals(
        "use_raw",
        "log",
        "num_categories",
        "categories_order",
        "title",
        "figsize",
        "gene_symbols",
        "var_group_positions",
        "var_group_labels",
        "var_group_rotation",
        "layer",
        "standard_scale",
        "ax",
        "values_df",
        "vmin",
        "vmax",
        "vcenter",
        "norm",
    )
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

        # Reserve width for group annotations (e.g., dendrogram or annotation tiles)
        self.group_extra_size = self._group_annotation_df.shape[1] if hasattr(self, "_group_annotation_df") else 0

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

    @property
    def group_extra_size(self):
        return (
            self._group_extra_size
            if hasattr(self, "_group_extra_size")
            else (self._group_annotation_df.shape[1] if self.has_annotations else 0)
        )

    @group_extra_size.setter
    def group_extra_size(self, value):
        self._group_extra_size = value

    @annotation_width.setter
    def annotation_width(self, value):
        self._annotation_width = value

    def make_figure(self) -> None:
        category_height = self.DEFAULT_CATEGORY_HEIGHT
        category_width = self.DEFAULT_CATEGORY_WIDTH

        if self.height is None:
            mainplot_height = len(self.categories) * category_height
            mainplot_width = len(self.var_names) * category_width + self.group_extra_size
            if self.are_axes_swapped:
                mainplot_height, mainplot_width = mainplot_width, mainplot_height

            height = mainplot_height + 1  # +1 for labels

            # if the number of categories is small use
            # a larger height, otherwise the legends do not fit
            self.height = max([self.min_figure_height, height])
            self.width = mainplot_width + self.legends_width
        else:
            self.min_figure_height = self.height
            mainplot_height = self.height

            mainplot_width = self.width - (self.legends_width + self.group_extra_size)

        return_ax_dict = {}
        # define a layout of 1 rows x 2 columns
        #   first ax is for the main figure.
        #   second ax is to plot legends
        legends_width_spacer = 0.7 / self.width

        self.fig, gs = make_grid_spec(
            self.ax or (self.width, self.height),
            nrows=1,
            ncols=2,
            wspace=legends_width_spacer,
            width_ratios=[mainplot_width + self.group_extra_size, self.legends_width],
        )

        if self.var_groups:
            # add some space in case 'brackets' want to be plotted on top of the image
            if self.are_axes_swapped:
                var_groups_height = category_height
            else:
                var_groups_height = category_height / 2

        else:
            var_groups_height = 0

        mainplot_width = mainplot_width - self.group_extra_size
        spacer_height = self.height - var_groups_height - mainplot_height
        if not self.are_axes_swapped:
            height_ratios = [spacer_height, var_groups_height, mainplot_height]
            width_ratios = [mainplot_width, self.group_extra_size]

        else:
            height_ratios = [spacer_height, self.group_extra_size, mainplot_height]
            width_ratios = [mainplot_width, var_groups_height]
            # gridspec is the same but rows and columns are swapped

        if self.fig_title is not None and self.fig_title.strip() != "":
            # for the figure title use the ax that contains
            # all the main graphical elements (main plot, dendrogram etc)
            # otherwise the title may overlay with the figure.
            # also, this puts the title centered on the main figure and not
            # centered between the main figure and the legends
            _ax = self.fig.add_subplot(gs[0, 0])
            _ax.axis("off")
            _ax.set_title(self.fig_title)

        # the main plot is divided into three rows and two columns
        # first row is an spacer that is adjusted in case the
        #           legends need more height than the main plot
        # second row is for brackets (if needed),
        # third row is for mainplot and dendrogram/totals (legend goes in gs[0,1]
        # defined earlier)
        # Add annotation column before main matrix if annotations are provided
        spacing = 0.3  # increase this value for more space between annotation and main plot
        ncols = 4 if self.has_annotations else 2
        width_ratios = (
            [self.annotation_width, spacing, mainplot_width, self.group_extra_size]
            if self.has_annotations
            else [mainplot_width, self.group_extra_size]
        )

        mainplot_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=3,
            ncols=ncols,
            wspace=self.wspace,
            hspace=0.0,
            subplot_spec=gs[0, 0],
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        # Create optional annotation axis
        if self.has_annotations:
            annot_ax = self.fig.add_subplot(mainplot_gs[2, 0])
            return_ax_dict["annotation_ax"] = annot_ax
            main_ax = self.fig.add_subplot(mainplot_gs[2, 2])
        else:
            main_ax = self.fig.add_subplot(mainplot_gs[2, 0])

        return_ax_dict["mainplot_ax"] = main_ax

        if not self.are_axes_swapped:
            if self.plot_group_extra is not None:
                group_extra_ax = self.fig.add_subplot(mainplot_gs[2, 1], sharey=main_ax)
                group_extra_orientation = "right"
            if self.var_groups:
                gene_groups_ax = self.fig.add_subplot(mainplot_gs[1, 0], sharex=main_ax)
                var_group_orientation = "top"
        else:
            if self.plot_group_extra:
                group_extra_ax = self.fig.add_subplot(mainplot_gs[1, 0], sharex=main_ax)
                group_extra_orientation = "top"
            if self.var_groups:
                gene_groups_ax = self.fig.add_subplot(mainplot_gs[2, 1], sharey=main_ax)
                var_group_orientation = "right"

        if self.plot_group_extra is not None:
            if self.plot_group_extra["kind"] == "dendrogram":
                _plot_dendrogram(
                    group_extra_ax,
                    self.adata,
                    self.groupby,
                    dendrogram_key=self.plot_group_extra["dendrogram_key"],
                    ticks=self.plot_group_extra["dendrogram_ticks"],
                    orientation=group_extra_orientation,
                )
            if self.plot_group_extra["kind"] == "group_totals":
                self._plot_totals(group_extra_ax, group_extra_orientation)

            return_ax_dict["group_extra_ax"] = group_extra_ax

        # plot group legends on top or left of main_ax (if given)
        if self.var_groups:
            _plot_var_groups_brackets(
                gene_groups_ax,
                var_groups=self.var_groups,
                rotation=self.var_group_rotation,
                left_adjustment=0.2,
                right_adjustment=0.7,
                orientation=var_group_orientation,
                wide=True,
            )
            return_ax_dict["gene_group_ax"] = gene_groups_ax

        # plot the mainplot
        normalize = self._mainplot(main_ax, annotation_ax=return_ax_dict.get("annotation_ax"))

        # code from pandas.plot in add_totals adds
        # minor ticks that need to be removed
        main_ax.yaxis.set_tick_params(which="minor", left=False, right=False)
        main_ax.xaxis.set_tick_params(which="minor", top=False, bottom=False, length=0)
        main_ax.set_zorder(100)
        if self.legends_width > 0:
            legend_ax = self.fig.add_subplot(gs[0, 1])
            self._plot_legend(legend_ax, return_ax_dict, normalize)

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

        # 2. Matrix plot in main_ax
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

        main_ax.tick_params(axis="both", labelsize="small")
        main_ax.grid(visible=False)
        main_ax.set_ylim(len(y_labels), 0)
        main_ax.set_xlim(0, len(x_labels))

        # 3. Optional: plot annotation tiles in annotation_ax
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


@old_positionals(
    "use_raw",
    "log",
    "num_categories",
    "figsize",
    "dendrogram",
    "title",
    "cmap",
    "colorbar_title",
    "gene_symbols",
    "var_group_positions",
    "var_group_labels",
    "var_group_rotation",
    "layer",
    "standard_scale",
    # 17 positionals are enough for backwards compatibility
)
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
    """Create a heatmap of the mean expression values per group of each var_names.

    This function provides a convenient interface to the :class:`~scanpy.pl.MatrixPlot`
    class. If you need more flexibility, you should use :class:`~scanpy.pl.MatrixPlot`
    directly.

    Parameters
    ----------
    {common_plot_args}
    {groupby_plots_args}
    {show_save_ax}
    {vminmax}
    kwds
        Are passed to :func:`matplotlib.pyplot.pcolor`.

    Returns
    -------
    If `return_fig` is `True`, returns a :class:`~scanpy.pl.MatrixPlot` object,
    else if `show` is false, return axes dict

    See Also
    --------
    :class:`~scanpy.pl.MatrixPlot`: The MatrixPlot class can be used to to control
        several visual parameters not available in this function.
    :func:`~scanpy.pl.rank_genes_groups_matrixplot`: to plot marker genes
        identified using the :func:`~scanpy.tl.rank_genes_groups` function.

    Examples
    --------

    .. plot::
        :context: close-figs

        import scanpy as sc
        adata = sc.datasets.pbmc68k_reduced()
        markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
        sc.pl.matrixplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Using var_names as dict:

    .. plot::
        :context: close-figs

        markers = {{'T-cell': 'CD3D', 'B-cell': 'CD79A', 'myeloid': 'CST3'}}
        sc.pl.matrixplot(adata, markers, groupby='bulk_labels', dendrogram=True)

    Get Matrix object for fine tuning:

    .. plot::
        :context: close-figs

        mp = sc.pl.matrixplot(adata, markers, 'bulk_labels', return_fig=True)
        mp.add_totals().style(edge_color='black').show()

    The axes used can be obtained using the get_axes() method

    .. plot::
        :context: close-figs

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
        **kwds,
    )

    if dendrogram:
        mp.add_dendrogram(dendrogram_key=_dk(dendrogram))
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
