import marsilea as ma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marsilea.plotter import Bar, Chunk, Labels

from ._palettes import default_palette


class BaseGSEA:
    """
    Base class for GSEA plotting.

    Handles data preparation, top-N selection, and color palette assignment.

    Args:
        enrichment_results (pd.DataFrame): GSEA results table.
        group_key (str or list): Column(s) to group by.
        top_n (int): Number of top terms per group to plot.
        adata (AnnData, optional): Annotated data object for color extraction.
        values (list, optional): List of group values for color mapping.
        colors (list, optional): List of colors corresponding to `values`.
    """

    def __init__(self, enrichment_results, group_key, top_n=5, adata=None, values=None, colors=None):
        self.df = enrichment_results.copy()
        self.group_key = group_key
        self.top_n = top_n
        self.adata = adata
        self.values = values
        self.colors = colors

        # Ensure group_cols is a list
        self.group_cols = [group_key] if isinstance(group_key, str) else group_key

        # Create a combined group column
        self.df["group"] = self.df[self.group_cols].astype(str).agg(" | ".join, axis=1)
        self.df["-log10(padj)"] = -np.log10(self.df["Adjusted P-value"])

        # Select top_n terms per group by smallest adjusted p-value
        self.df_top = self.df.groupby("group", group_keys=False).apply(
            lambda g: g.nsmallest(self.top_n, "Adjusted P-value")
        )

        self.groups = pd.Categorical(self.df_top["group"].unique().tolist())
        self.palette = self._make_palette()

    def _make_palette(self):
        """
        Create a color palette for each group.

        Returns
        -------
            dict: Mapping from group name to color.
        """
        group_names = list(self.groups.categories)
        palette = {}

        if self.adata is not None:
            # Try to extract colors from AnnData object
            for g in group_names:
                try:
                    key_vals = g.split(" | ")
                    colors_for_each = []
                    for col, val in zip(self.group_cols, key_vals, strict=False):
                        if col in self.adata.obs and f"{col}_colors" in self.adata.uns:
                            values = self.adata.obs[col].cat.categories
                            adata_palette = dict(zip(values, self.adata.uns[f"{col}_colors"], strict=False))
                            colors_for_each.append(adata_palette.get(val))
                    palette[g] = next(c for c in colors_for_each if c is not None)
                except StopIteration:
                    palette[g] = None
            # Assign default colors for missing
            missing = [k for k, v in palette.items() if v is None]
            for k, color in zip(missing, default_palette(len(missing)), strict=False):
                palette[k] = color
        elif self.values is not None and self.colors is not None:
            # Use provided values/colors mapping
            palette = dict(zip(self.values, self.colors, strict=False))
            for g in group_names:
                palette.setdefault(g, default_palette(1)[0])
        else:
            # Use default palette
            palette = dict(zip(group_names, default_palette(len(group_names)), strict=False))
        return palette


class GSEAHeatmap(BaseGSEA):
    """
    GSEA Heatmap Plotter.

    Visualizes GSEA results as a heatmap of -log10(adjusted p-value).
    """

    def plot(self, figsize=(6, 6), show=True):
        """
        Plot the GSEA heatmap.

        Args:
            figsize (tuple): Figure size (width, height).
            show (bool): Whether to render the plot immediately.

        Returns
        -------
            marsilea.Heatmap: The heatmap plot object.
        """
        matrix = self.df_top.pivot_table(index="Term", columns="group", values="-log10(padj)", aggfunc="first")

        anno = Chunk(matrix.columns.tolist(), [self.palette.get(g) for g in matrix.columns], padding=10)
        labels = Labels(matrix.index.tolist())

        cb = ma.Heatmap(matrix, width=figsize[0], height=figsize[1], cbar_kws={"title": "-log10(padj)"})
        cb.group_cols(pd.Categorical(matrix.columns.tolist()))
        cb.add_top(anno, pad=0.05)
        cb.add_left(labels, pad=0.05)
        cb.add_legends()

        if show:
            with plt.rc_context(rc={"axes.grid": False, "grid.color": ".8"}):
                cb.render()
        return cb


class GSEADot(BaseGSEA):
    """
    GSEA Dot Plot Plotter.

    Visualizes GSEA results as a dot plot, where color encodes -log10(adjusted p-value)
    and size encodes Odds Ratio.
    """

    def plot(self, figsize=(6, 6), scale=1, show=True):
        """
        Plot the GSEA dot plot.

        Args:
            figsize (tuple): Figure size (width, height).
            scale (float): Scaling factor for dot sizes.
            show (bool): Whether to render the plot immediately.

        Returns
        -------
            marsilea.SizedHeatmap: The dot plot object.
        """
        matrix_color = self.df_top.pivot_table(index="Term", columns="group", values="-log10(padj)", aggfunc="first")
        matrix_size = self.df_top.pivot_table(
            index="Term", columns="group", values="Odds Ratio", aggfunc="first"
        ).fillna(0)

        anno = Chunk(matrix_color.columns.tolist(), [self.palette.get(g) for g in matrix_color.columns], padding=10)
        labels = Labels(matrix_color.index.tolist())

        cb = ma.SizedHeatmap(
            color=matrix_color,
            size=matrix_size,
            sizes=(1, scale * 100),
            width=figsize[0],
            height=figsize[1],
            color_legend_kws={"title": "-log10(padj)"},
            size_legend_kws={"title": "Odds Ratio"},
        )
        cb.group_cols(pd.Categorical(matrix_color.columns.tolist()))
        cb.add_top(anno, pad=0.05)
        cb.add_left(labels, pad=0.05)
        cb.add_legends()
        if show:
            with plt.rc_context(rc={"axes.grid": False, "grid.color": ".8"}):
                cb.render()
        return cb


def gsea_heatmap(
    enrichment_results, group_key, top_n=5, adata=None, values=None, colors=None, figsize=(6, 6), show=True
):
    """
    Convenience function to plot a GSEA heatmap.

    Args:
        enrichment_results (pd.DataFrame): GSEA results table.
        group_key (str or list): Column(s) to group by.
        top_n (int): Number of top terms per group to plot.
        adata (AnnData, optional): Annotated data object for color extraction.
        values (list, optional): List of group values for color mapping.
        colors (list, optional): List of colors corresponding to `values`.
        figsize (tuple): Figure size (width, height).
        show (bool): Whether to render the plot immediately.

    Returns
    -------
        marsilea.Heatmap: The heatmap plot object.
    """
    plotter = GSEAHeatmap(enrichment_results, group_key, top_n, adata, values, colors)
    return plotter.plot(figsize=figsize, show=show)


def gsea_dotplot(
    enrichment_results, group_key, top_n=5, adata=None, values=None, colors=None, figsize=(6, 6), show=True
):
    """
    Convenience function to plot a GSEA dot plot.

    Args:
        enrichment_results (pd.DataFrame): GSEA results table.
        group_key (str or list): Column(s) to group by.
        top_n (int): Number of top terms per group to plot.
        adata (AnnData, optional): Annotated data object for color extraction.
        values (list, optional): List of group values for color mapping.
        colors (list, optional): List of colors corresponding to `values`.
        figsize (tuple): Figure size (width, height).
        show (bool): Whether to render the plot immediately.

    Returns
    -------
        marsilea.SizedHeatmap: The dot plot object.
    """
    plotter = GSEADot(enrichment_results, group_key, top_n, adata, values, colors)
    return plotter.plot(figsize=figsize, show=show)


def gsea_barplot(
    enrichment_results: pd.DataFrame,
    group_key: str | list[str],
    top_n: int = 5,
    adata=None,
    values=None,
    colors=None,
    figsize=(4, 5),
    show: bool = True,
) -> ma.ClusterBoard:
    """
    Create a horizontal bar plot of top GSEA enrichment terms per group using Marsilea.

    Supports grouping by multiple keys.

    Parameters
    ----------
    enrichment_results : pd.DataFrame
        DataFrame with columns including: "Term", "Adjusted P-value", and the group keys.
    group_key : str or list
        Single column or list of column names to group by.
    top_n : int
        Number of top terms to show per group.
    adata : AnnData or None
        For extracting palette by group.
    values : list or None
        Group names, if no AnnData is given.
    colors : list or None
        Colors corresponding to group names.
    figsize : tuple
        Width and height in inches.
    show : bool
        Whether to render the plot.

    Returns
    -------
    marsilea.ClusterBoard
        The ClusterBoard object.
    """
    df = enrichment_results.copy()
    if isinstance(group_key, str):
        group_cols = [group_key]
    else:
        group_cols = group_key

    df["group"] = df[group_cols].astype(str).agg(" | ".join, axis=1)
    df["-log10(padj)"] = -np.log10(df["Adjusted P-value"])

    # Top N per group
    df_top = df.groupby("group", group_keys=False).apply(lambda g: g.nsmallest(top_n, "Adjusted P-value"))
    df_top = df_top.set_index("Term")

    group = pd.Categorical(df_top["group"].tolist())

    # Build palette
    group_names = list(group.categories)
    if adata is not None:
        # Try to fetch matching adata palette entries if available
        palette = {}
        for g in group_names:
            try:
                key_vals = g.split(" | ")
                colors_for_each = []
                for col, val in zip(group_cols, key_vals, strict=False):
                    if col in adata.obs and f"{col}_colors" in adata.uns:
                        values = adata.obs[col].cat.categories
                        adata_palette = dict(zip(values, adata.uns[f"{col}_colors"], strict=False))
                        colors_for_each.append(adata_palette.get(val))
                # Take first non-None color
                palette[g] = next(c for c in colors_for_each if c is not None)
            except Exception:
                palette[g] = None
        # Replace None with defaults
        missing = [k for k, v in palette.items() if v is None]
        default_colors = default_palette(len(missing))
        for k, color in zip(missing, default_colors, strict=False):
            palette[k] = color
    elif values is not None and colors is not None:
        palette = dict(zip(values, colors, strict=False))
        for g in group_names:
            palette.setdefault(g, default_palette(1)[0])
    else:
        # Fall back to default palette
        palette = dict(zip(group_names, default_palette(len(group_names)), strict=False))

    # Plotting
    anno = Chunk(list(palette.keys()), list(palette.values()), padding=10)
    plot = Bar(
        df_top[["-log10(padj)"]].T, orient="h", label="-log10(padj)", group_kws={"color": list(palette.values())}
    )
    labels = Labels(list(df_top.index))

    cb = ma.ClusterBoard(df_top[["-log10(padj)"]], width=figsize[0], height=figsize[1])
    cb.add_layer(plot)
    cb.group_rows(group)
    cb.add_left(anno)
    cb.add_left(labels, pad=0.05)

    if show:
        with plt.rc_context(rc={"axes.grid": False, "grid.color": ".8"}):
            cb.render()
    return cb
