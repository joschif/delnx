import marsilea as ma
import numpy as np
import pandas as pd
from marsilea.plotter import Bar, Chunk, Labels

from ._palettes import default_palette


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
        cb.render()
    return cb
