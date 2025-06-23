"""Plotting functions for AnnData."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from itertools import pairwise, product
from types import NoneType
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colormaps, gridspec, patheffects, rcParams
from matplotlib import pyplot as plt
from matplotlib.colors import is_color_like
from packaging.version import Version
from pandas.api.types import CategoricalDtype, is_numeric_dtype

from .. import _logger as logg
from .. import get
from .._compat import CSBase
from .._settings import settings
from .._utils import (
    _check_use_raw,
    _empty,
    get_literal_vals,
    sanitize_anndata,
)
from . import _utils
from ._utils import (
    _deprecated_scale,
    _dk,
    check_colornorm,
    scatter_base,
    scatter_group,
    setup_axes,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal, Self

    from anndata import AnnData
    from cycler import Cycler
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, ListedColormap, Normalize
    from numpy.typing import NDArray
    from seaborn import FacetGrid
    from seaborn.matrix import ClusterGrid

    from .._utils import Empty
    from ._utils import (
        ColorLike,
        DensityNorm,
        _FontSize,
        _FontWeight,
        _LegendLoc,
    )

    # TODO: is that all?
    _Basis = Literal["pca", "tsne", "umap", "diffmap", "draw_graph_fr"]
    _VarNames = str | Sequence[str]


class VarGroups(NamedTuple):
    labels: Sequence[str]
    """Var labels."""
    positions: Sequence[tuple[int, int]]
    """Var positions.

    Each item in the list should contain the start and end position that the bracket should cover.
    Eg. `[(0, 4), (5, 8)]` means that there are two brackets,
    one for the var_names (eg genes) in positions 0-4 and other for positions 5-8
    """

    @classmethod
    def validate(cls, labels: Sequence[str] | None, positions: Sequence[tuple[int, int]] | None) -> Self | None:
        if labels is None and positions is None:
            return None
        if labels is None or positions is None:
            msg = "If var_group_labels or var_group_positions are given, both have to be given."
            raise ValueError(msg)
        if len(labels) != len(positions):
            msg = (
                "var_group_labels and var_group_positions must have the same length. "
                f"Got {len(labels)=} and {len(positions)=}."
            )
            raise ValueError(msg)
        return None if len(labels) == 0 else cls(labels, positions)


def _prepare_dataframe(
    adata: AnnData,
    var_names: _VarNames | Mapping[str, _VarNames],
    groupby: str | Sequence[str] | None = None,
    *,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    layer: str | None = None,
    gene_symbols: str | None = None,
) -> tuple[Sequence[str], pd.DataFrame]:
    """Prepare a data frame of categories (`groupby`) × `var_names`."""
    sanitize_anndata(adata)
    use_raw = _check_use_raw(adata, use_raw, layer=layer)
    if isinstance(var_names, str):
        var_names = [var_names]

    groupby_index = None
    if groupby is not None:
        if isinstance(groupby, str):
            groupby = [groupby]
        for group in groupby:
            if group not in [*adata.obs_keys(), adata.obs.index.name]:
                msg = f' or index name "{adata.obs.index.name}"' if adata.obs.index.name else ""
                raise ValueError(
                    f"groupby has to be a valid observation. "
                    f"Given {group}, is not in observations: {adata.obs_keys()}" + msg
                )
            if group in adata.obs.columns and group == adata.obs.index.name:
                raise ValueError(f"Given group {group} is both an index and a column, which is ambiguous.")
            if group == adata.obs.index.name:
                groupby_index = group

    if groupby_index is not None:
        groupby = groupby.copy()
        groupby.remove(groupby_index)

    keys = list(groupby or []) + list(np.unique(var_names))
    obs_tidy = get.obs_df(adata, keys=keys, layer=layer, use_raw=use_raw, gene_symbols=gene_symbols)
    assert np.all(np.array(keys) == np.array(obs_tidy.columns))

    if groupby_index is not None:
        obs_tidy.reset_index(inplace=True)
        groupby.append(groupby_index)

    if groupby is None:
        categorical = pd.Series(np.repeat("", len(obs_tidy))).astype("category")
    elif len(groupby) == 1 and is_numeric_dtype(obs_tidy[groupby[0]]):
        categorical = pd.cut(obs_tidy[groupby[0]], num_categories)
    elif len(groupby) == 1:
        categorical = obs_tidy[groupby[0]].astype("category")
        categorical.name = groupby[0]
    else:
        # Join values using " | " and preserve category order
        for g in groupby:
            obs_tidy[g] = obs_tidy[g].astype("category")
        from itertools import product

        # Compute joined category names and expected order
        compound_names = (obs_tidy[groupby].astype(str).agg(" | ".join, axis=1)).astype("category")
        product_order = [" | ".join(p) for p in product(*(obs_tidy[g].cat.categories for g in groupby))]
        observed_categories = compound_names.cat.categories.tolist()
        product_order_filtered = [cat for cat in product_order if cat in observed_categories]
        compound_names = compound_names.cat.reorder_categories(product_order_filtered)
        categorical = compound_names

    obs_tidy = obs_tidy[var_names].set_index(categorical)
    categories = obs_tidy.index.categories

    if log:
        obs_tidy = np.log1p(obs_tidy)

    return categories, obs_tidy



class _ReorderCats(TypedDict):
    categories_idx_ordered: Sequence[int]
    categories_ordered: Sequence[str]
    var_names_idx_ordered: Sequence[int] | None
    var_names_ordered: Sequence[str] | None
    var_groups: VarGroups | None


def _reorder_categories_after_dendrogram(
    adata: AnnData,
    groupby: str | Sequence[str],
    *,
    dendrogram_key: str | None,
    var_names: Sequence[str],
    var_groups: VarGroups | None,
    categories: Sequence[str],
) -> _ReorderCats:
    """Reorder the the groupby observations based on the dendrogram results.

    The function checks if a dendrogram has already been precomputed.
    If not, `sc.tl.dendrogram` is run with default parameters.

    The results found in `.uns[dendrogram_key]` are used to reorder `var_groups`.
    """
    if isinstance(groupby, str):
        groupby = [groupby]

    dendro_info = adata.uns[_get_dendrogram_key(adata, dendrogram_key, groupby, validate_groupby=True)]

    if categories is None:
        categories = adata.obs[dendro_info["groupby"]].cat.categories

    # order of groupby categories
    categories_idx_ordered = dendro_info["categories_idx_ordered"]
    categories_ordered = dendro_info["categories_ordered"]

    if len(categories) != len(categories_idx_ordered):
        msg = (
            "Incompatible observations. Dendrogram data has "
            f"{len(categories_idx_ordered)} categories but current groupby "
            f"observation {groupby!r} contains {len(categories)} categories. "
            "Most likely the underlying groupby observation changed after the "
            "initial computation of `sc.tl.dendrogram`. "
            "Please run `sc.tl.dendrogram` again.'"
        )
        raise ValueError(msg)

    # reorder var_groups (if any)
    if var_groups is None:
        var_names_idx_ordered = None
    elif set(var_groups.labels) == set(categories):
        positions_ordered = []
        labels_ordered = []
        position_start = 0
        var_names_idx_ordered = []
        for cat_name in categories_ordered:
            idx = var_groups.labels.index(cat_name)
            position = var_groups.positions[idx]
            _var_names = var_names[position[0] : position[1] + 1]
            var_names_idx_ordered.extend(range(position[0], position[1] + 1))
            positions_ordered.append((position_start, position_start + len(_var_names) - 1))
            position_start += len(_var_names)
            labels_ordered.append(var_groups.labels[idx])
        var_groups = VarGroups(labels_ordered, positions_ordered)
    else:
        logg.warning(
            "Groups are not reordered because the `groupby` categories "
            "and the `var_group_labels` are different.\n"
            f"categories: {_format_first_three_categories(categories)}\n"
            f"var_group_labels: {_format_first_three_categories(var_groups.labels)}"
        )
        var_names_idx_ordered = list(range(len(var_names)))

    if var_names_idx_ordered is not None:
        var_names_ordered = [var_names[x] for x in var_names_idx_ordered]
    else:
        var_names_ordered = None

    return _ReorderCats(
        categories_idx_ordered=categories_idx_ordered,
        categories_ordered=dendro_info["categories_ordered"],
        var_names_idx_ordered=var_names_idx_ordered,
        var_names_ordered=var_names_ordered,
        var_groups=var_groups,
    )


def _format_first_three_categories(categories):
    """Clean up warning message."""
    categories = list(categories)
    if len(categories) > 3:
        categories = categories[:3] + ["etc."]
    return ", ".join(categories)


def _get_dendrogram_key(
    adata: AnnData,
    dendrogram_key: str | None,
    groupby: str | Sequence[str],
    *,
    validate_groupby: bool = False,
) -> str:
    # the `dendrogram_key` can be a bool an NoneType or the name of the
    # dendrogram key. By default the name of the dendrogram key is 'dendrogram'
    if dendrogram_key is None:
        if isinstance(groupby, str):
            dendrogram_key = f"dendrogram_{groupby}"
        elif isinstance(groupby, Sequence):
            dendrogram_key = f"dendrogram_{'_'.join(groupby)}"
        else:
            msg = f"groupby has wrong type: {type(groupby).__name__}."
            raise AssertionError(msg)

    if dendrogram_key not in adata.uns:
        from ..tl._dendrogram import dendrogram

        logg.warning(
            f"dendrogram data not found (using key={dendrogram_key}). "
            "Running `sc.tl.dendrogram` with default parameters. For fine "
            "tuning it is recommended to run `sc.tl.dendrogram` independently."
        )
        dendrogram(adata, groupby, key_added=dendrogram_key)

    if "dendrogram_info" not in adata.uns[dendrogram_key]:
        msg = f"The given dendrogram key ({dendrogram_key!r}) does not contain valid dendrogram information."
        raise ValueError(msg)

    if validate_groupby:
        existing_groupby = adata.uns[dendrogram_key]["groupby"]
        if groupby != existing_groupby:
            msg = (
                "Incompatible observations. The precomputed dendrogram contains "
                f"information for the observation: {groupby!r} while the plot is "
                f"made for the observation: {existing_groupby!r}. "
                "Please run `sc.tl.dendrogram` using the right observation.'"
            )
            raise ValueError(msg)

    return dendrogram_key


def _plot_dendrogram(
    dendro_ax: Axes,
    adata: AnnData,
    groupby: str | Sequence[str],
    *,
    dendrogram_key: str | None = None,
    orientation: Literal["top", "bottom", "left", "right"] = "right",
    remove_labels: bool = True,
    ticks: Collection[float] | None = None,
):
    """Plot a dendrogram on the given ax.

    Uses the precomputed dendrogram information stored in `.uns[dendrogram_key]`.
    """
    dendrogram_key = _get_dendrogram_key(adata, dendrogram_key, groupby)

    def translate_pos(pos_list, new_ticks, old_ticks):
        """Transform the dendrogram coordinates to a given new position.

        The xlabel_pos and orig_ticks should be of the same length.

        This is mostly done for the heatmap case, where the position of the
        dendrogram leaves needs to be adjusted depending on the category size.

        Parameters
        ----------
        pos_list
            list of dendrogram positions that should be translated
        new_ticks
            sorted list of goal tick positions (e.g. [0,1,2,3] )
        old_ticks
            sorted list of original tick positions (e.g. [5, 15, 25, 35]),
            This list is usually the default position used by
            `scipy.cluster.hierarchy.dendrogram`.

        Returns
        -------
        translated list of positions

        Examples
        --------
        >>> translate_pos(
        ...     [5, 15, 20, 21],
        ...     [0, 1, 2, 3],
        ...     [5, 15, 25, 35],
        ... )
        [0, 1, 1.5, 1.6]

        """
        # of given coordinates.

        if not isinstance(old_ticks, list):
            # assume that the list is a numpy array
            old_ticks = old_ticks.tolist()
        new_xs = []
        for x_val in pos_list:
            if x_val in old_ticks:
                new_x_val = new_ticks[old_ticks.index(x_val)]
            else:
                # find smaller and bigger indices
                idx_next = np.searchsorted(old_ticks, x_val, side="left")
                idx_prev = idx_next - 1
                old_min = old_ticks[idx_prev]
                old_max = old_ticks[idx_next]
                new_min = new_ticks[idx_prev]
                new_max = new_ticks[idx_next]
                new_x_val = ((x_val - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            new_xs.append(new_x_val)
        return new_xs

    dendro_info = adata.uns[dendrogram_key]["dendrogram_info"]
    leaves = dendro_info["ivl"]
    icoord = np.array(dendro_info["icoord"])
    dcoord = np.array(dendro_info["dcoord"])

    orig_ticks = np.arange(5, len(leaves) * 10 + 5, 10).astype(float)
    # check that ticks has the same length as orig_ticks
    if ticks is not None and len(orig_ticks) != len(ticks):
        logg.warning("ticks argument does not have the same size as orig_ticks. The argument will be ignored")
        ticks = None

    for xs, ys in zip(icoord, dcoord, strict=True):
        if ticks is not None:
            xs = translate_pos(xs, ticks, orig_ticks)
        if orientation in ["right", "left"]:
            xs, ys = ys, xs
        dendro_ax.plot(xs, ys, color="black")

    dendro_ax.tick_params(bottom=False, top=False, left=False, right=False)
    ticks = ticks if ticks is not None else orig_ticks
    if orientation in ["right", "left"]:
        dendro_ax.set_yticks(ticks)
        dendro_ax.set_yticklabels(leaves, fontsize="small", rotation=0)
        dendro_ax.tick_params(labelbottom=False, labeltop=False)
        if orientation == "left":
            xmin, xmax = dendro_ax.get_xlim()
            dendro_ax.set_xlim(xmax, xmin)
            dendro_ax.tick_params(labelleft=False, labelright=True)
    else:
        dendro_ax.set_xticks(ticks)
        dendro_ax.set_xticklabels(leaves, fontsize="small", rotation=90)
        dendro_ax.tick_params(labelleft=False, labelright=False)
        if orientation == "bottom":
            ymin, ymax = dendro_ax.get_ylim()
            dendro_ax.set_ylim(ymax, ymin)
            dendro_ax.tick_params(labeltop=True, labelbottom=False)

    if remove_labels:
        dendro_ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    dendro_ax.grid(visible=False)

    dendro_ax.spines["right"].set_visible(False)
    dendro_ax.spines["top"].set_visible(False)
    dendro_ax.spines["left"].set_visible(False)
    dendro_ax.spines["bottom"].set_visible(False)


def _plot_categories_as_colorblocks(
    groupby_ax: Axes,
    obs_tidy: pd.DataFrame,
    colors=None,
    orientation: Literal["top", "bottom", "left", "right"] = "left",
    cmap_name: str = "tab20",
):
    """Plot categories as colored blocks.

    If orientation is 'left', the categories are plotted vertically,
    otherwise they are plotted horizontally.

    Parameters
    ----------
    groupby_ax
    obs_tidy
    colors
        Sequence of valid color names to use for each category.
    orientation
    cmap_name
        Name of colormap to use, in case colors is None

    Returns
    -------
    ticks position, labels, colormap

    """
    groupby = obs_tidy.index.name
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if colors is None:
        groupby_cmap = colormaps.get_cmap(cmap_name)
    else:
        groupby_cmap = ListedColormap(colors, groupby + "_cmap")
    norm = BoundaryNorm(np.arange(groupby_cmap.N + 1) - 0.5, groupby_cmap.N)

    # determine groupby label positions such that they appear
    # centered next/below to the color code rectangle assigned to the category
    value_sum = 0
    ticks = []  # list of centered position of the labels
    labels = []
    label2code = {}  # dictionary of numerical values asigned to each label
    for code, (label, value) in enumerate(obs_tidy.index.value_counts(sort=False).items()):
        ticks.append(value_sum + (value / 2))
        labels.append(label)
        value_sum += value
        label2code[label] = code

    groupby_ax.grid(visible=False)

    if orientation == "left":
        groupby_ax.imshow(
            np.array([[label2code[lab] for lab in obs_tidy.index]]).T,
            aspect="auto",
            cmap=groupby_cmap,
            norm=norm,
        )
        if len(labels) > 1:
            groupby_ax.set_yticks(ticks)
            groupby_ax.set_yticklabels(labels)

        # remove y ticks
        groupby_ax.tick_params(axis="y", left=False, labelsize="small")
        # remove x ticks and labels
        groupby_ax.tick_params(axis="x", bottom=False, labelbottom=False)

        # remove surrounding lines
        groupby_ax.spines["right"].set_visible(False)
        groupby_ax.spines["top"].set_visible(False)
        groupby_ax.spines["left"].set_visible(False)
        groupby_ax.spines["bottom"].set_visible(False)

        groupby_ax.set_ylabel(groupby)
    else:
        groupby_ax.imshow(
            np.array([[label2code[lab] for lab in obs_tidy.index]]),
            aspect="auto",
            cmap=groupby_cmap,
            norm=norm,
        )
        if len(labels) > 1:
            groupby_ax.set_xticks(ticks)
            # if the labels are small do not rotate them
            rotation = 0 if max(len(str(x)) for x in labels) < 3 else 90
            groupby_ax.set_xticklabels(labels, rotation=rotation)

        # remove x ticks
        groupby_ax.tick_params(axis="x", bottom=False, labelsize="small")
        # remove y ticks and labels
        groupby_ax.tick_params(axis="y", left=False, labelleft=False)

        # remove surrounding lines
        groupby_ax.spines["right"].set_visible(False)
        groupby_ax.spines["top"].set_visible(False)
        groupby_ax.spines["left"].set_visible(False)
        groupby_ax.spines["bottom"].set_visible(False)

        groupby_ax.set_xlabel(groupby)

    return label2code, ticks, labels, groupby_cmap, norm


def _plot_colorbar(mappable, fig, subplot_spec, max_cbar_height: float = 4.0):
    """Plot a vertical color bar based on mappable.

    The height of the colorbar is min(figure-height, max_cmap_height).

    Parameters
    ----------
    mappable
        The image to which the colorbar applies.
    fig
        The figure object
    subplot_spec
        The gridspec subplot. Eg. axs[1,2]
    max_cbar_height
        The maximum colorbar height

    Returns
    -------
    color bar ax

    """
    width, height = fig.get_size_inches()
    if height > max_cbar_height:
        # to make the colorbar shorter, the
        # ax is split and the lower portion is used.
        axs2 = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=subplot_spec,
            height_ratios=[height - max_cbar_height, max_cbar_height],
        )
        heatmap_cbar_ax = fig.add_subplot(axs2[1])
    else:
        heatmap_cbar_ax = fig.add_subplot(subplot_spec)
    plt.colorbar(mappable, cax=heatmap_cbar_ax)
    return heatmap_cbar_ax


def _check_var_names_type(
    var_names: _VarNames | Mapping[str, _VarNames],
    var_group_labels: Sequence[str] | None = None,
    var_group_positions: Sequence[tuple[int, int]] | None = None,
) -> tuple[Sequence[str], VarGroups | None]:
    """If var_names is a dict, set the `var_group_labels` and `var_group_positions`.

    Returns
    -------
    var_names, var_groups

    """
    from ._baseplot_class import _var_groups

    if isinstance(var_names, Mapping):
        return _var_groups(var_names)

    if isinstance(var_names, str):
        var_names = [var_names]
    return var_names, VarGroups.validate(var_group_labels, var_group_positions)
