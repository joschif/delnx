import marsilea as ma
import marsilea.plotter as mp
import numpy as np
import pandas as pd


def gsea_barplot(
    enrichment_results: pd.DataFrame,
    group_key: str,
    top_n: int = 5,
    adata=None,
    values=None,
    colors=None,
    figsize=(4, 5),
    show: bool = True,
) -> ma.ClusterBoard:
    """
    Create a horizontal bar plot of top GSEA enrichment terms per group using Marsilea.

    Parameters
    ----------
    enrichment_results : pd.DataFrame
        DataFrame containing GSEA results with at least the columns: "Term", "Adjusted P-value", and group_key.
    group_key : str, default="group"
        Column name in enrichment_results to group by (e.g., cell type).
    top_n : int, default=5
        Number of top enriched terms to include for each group.
    adata : AnnData or None, default=None
        AnnData object to extract palette from (using group_key).
    values : list or None, default=None
        List of group names (used if adata is not provided).
    colors : list or None, default=None
        List of colors corresponding to values (used if adata is not provided).
    figsize : tuple, default=(4, 5)
        Size of the figure (not used directly, for compatibility).
    show : bool, default=True
        If True, renders the plot.

    Returns
    -------
    marsilea.ClusterBoard
        The Marsilea ClusterBoard object.
    """
    df = enrichment_results[["Term", "Adjusted P-value", group_key]].copy()
    df["-log10(padj)"] = -np.log10(df["Adjusted P-value"])
    df = df.groupby(group_key, group_keys=False).apply(lambda g: g.nsmallest(top_n, "Adjusted P-value"))
    df = df.set_index("Term")
    group = pd.Categorical(df[group_key].tolist())

    # Build palette
    if adata is not None:
        key = group_key
        values = adata.obs[key].cat.categories
        palette = adata.uns.get(f"{key}_colors")
        palette = dict(zip(values, palette, strict=False))
        palette = {e: palette[e] for e in list(group.categories)}
    elif values is not None and colors is not None:
        palette = dict(zip(values, colors, strict=False))
        palette = {e: palette[e] for e in list(group.categories)}
    else:
        raise ValueError("Either adata or both values and colors must be provided to build the palette.")

    anno = ma.plotter.Chunk(list(palette.keys()), list(palette.values()), padding=10)
    plot = mp.Bar(df[["-log10(padj)"]].T, orient="h", label="-log10(padj)", group_kws={"color": list(palette.values())})
    labels = mp.Labels(list(df.index))

    cb = ma.ClusterBoard(df[["-log10(padj)"]], width=figsize[0], height=figsize[1])
    cb.add_layer(plot)
    cb.group_rows(group)
    cb.add_left(anno)
    cb.add_left(labels, pad=0.05)
    if show:
        cb.render()
    return cb
