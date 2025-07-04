import numpy as np
import scanpy as sc
from anndata import AnnData


def filter_genes(
    adata: AnnData,
    mode: str = "quantile",
    quantile: float = 0.25,
    min_total_count: int = 100,
    layer: str | None = "counts",
) -> np.ndarray:
    """
    Filter genes in an AnnData object based on total count thresholds.

    Parameters
    ----------
    adata : AnnData
        Single-cell AnnData object.
    mode : str, default "quantile"
        Filtering mode: 'quantile' or 'absolute'.
    quantile : float, default 0.25
        If mode='quantile': keep genes above this quantile of total counts.
    min_total_count : int, default 100
        If mode='absolute': keep genes with total counts >= this value.
    layer : str or None, default "counts"
        Which layer to use for counts. If None, use adata.X.

    Returns
    -------
    np.ndarray
        Boolean mask (n_genes,) indicating which genes to keep.
    """
    # Select count matrix
    counts = adata.layers[layer] if layer else adata.X
    if not isinstance(counts, np.ndarray):
        counts = counts.toarray()

    # Compute total count per gene
    total_counts = counts.sum(axis=0)

    if mode == "quantile":
        threshold = np.quantile(total_counts, quantile)
        return total_counts > threshold

    if mode == "absolute":
        keep_mask, _ = sc.pp.filter_genes(adata, min_counts=min_total_count, inplace=False)
        return keep_mask

    raise ValueError(f"Unknown mode '{mode}'. Choose 'quantile' or 'absolute'.")
