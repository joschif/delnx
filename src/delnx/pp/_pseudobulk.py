"""Pseudobulking of single-cell RNA-seq data.

This module provides functionality to aggregate single-cell RNA-seq data into pseudobulk
samples, which can improve statistical power for differential expression analysis.
Pseudobulking aggregates expression profiles from multiple cells within the same sample
and cell type/group, reducing the impact of technical noise and dropout events.

The implementation leverages the decoupler package for efficient data aggregation
with support for various aggregation methods (sum, mean, median) and optional
creation of pseudoreplicates.
"""

import decoupler as dc
import numpy as np
from anndata import AnnData


def pseudobulk(
    adata: AnnData,
    sample_key: str = "batch",
    group_key: str | None = None,
    n_pseudoreps: int | None = None,
    layer: str | None = None,
    min_cells: int | None = 5,
    min_counts: int | None = 5000,
    mode: str = "sum",
    **kwargs,
) -> AnnData:
    """Create pseudobulk samples from single-cell RNA-seq data.

    This function aggregates single-cell RNA-seq data into pseudobulk samples based on specified sample and group identifiers. It can optionally create random pseudoreplicates.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        AnnData object containing single-cell expression data.
    sample_key : str, default="batch"
        Column name in `adata.obs` that contains the sample identifiers.
        Each unique value will become a separate pseudobulk sample
        (or multiple samples if `n_pseudoreps` is specified).
    group_key : str | None, default=None
        Column name in `adata.obs` that contains group identifiers like cell types.
        If provided, pseudobulk samples will be generated separately for each combination
        of sample and group, enabling cell type-specific analysis.
    n_pseudoreps : int | None, default=None
        Number of pseudoreplicates to create per sample. If :obj:`None`, uses the original
        sample structure without resampling. If specified, creates `n_pseudoreps`
        pseudoreplicates per sample by randomly sampling cells with replacement.
    layer : str | None, default=None
        Layer in `adata.layers` to use for aggregation. If :obj:`None`, uses `adata.X`.
        Should contain raw or normalized counts, not log-transformed values.
    mode : str, default="sum"
        Method to aggregate cell-level data into pseudobulk samples:
            - "sum": Sum of counts (recommended for RNA-seq data)
            - "mean": Average of counts across cells
    min_cells : int | None, default=5
        Minimum number of cells required in a pseudobulk sample to retain it.
        Samples with fewer cells will be discarded.
    min_counts : int | None, default=5000
        Minimum total counts required in a pseudobulk sample to retain it.
        Samples with fewer total counts will be discarded.
    **kwargs
        Additional arguments to pass to :func:`decoupler.pp.pseudobulk`

    Returns
    -------
        :class:`~anndata.AnnData` object containing the pseudobulk data. The structure changes from
        cell-level to sample-level, with each row representing a pseudobulk sample.
        Original sample and group identifiers are preserved in the observations.

    Notes
    -----
    - Wrapper around the decoupler (https://github.com/scverse/decoupler) pseudobulk function to support pseudo-replicates
    - It's generally recommended to aggregate counts using the "sum" mode and then re-normalize rather than using "mean" directly on log-normalized data.

    Examples
    --------
    Basic pseudobulking by sample:

    >>> import scanpy as sc
    >>> import delnx as dx
    >>> adata = sc.read_h5ad("single_cell_data.h5ad")
    >>> # Assuming adata.obs["sample"] contains sample identifiers
    >>> pseudobulk_adata = dx.pp.pseudobulk(adata, sample_key="sample")

    Pseudobulking by sample and cell type:

    >>> # Create cell type-specific pseudobulk samples
    >>> pseudobulk_adata = dx.pp.pseudobulk(adata, sample_key="sample", group_key="cell_type")

    Creating pseudoreplicates for assessing technical variation:

    >>> # Generate 5 pseudoreplicates per original sample
    >>> pseudobulk_adata = dx.pp.pseudobulk(adata, sample_key="sample", n_pseudoreps=5)
    """
    # Create pseudoreplicates if requested
    if n_pseudoreps is not None:
        # Randomly assign each cell to one of the n_pseudoreps replicates for its sample
        pseudoreps = np.random.choice(
            np.arange(n_pseudoreps),
            size=adata.n_obs,
            replace=True,
        )
        # Create unique identifiers for each pseudoreplicate by combining sample ID with replicate number
        adata.obs["psbulk_replicate"] = adata.obs[sample_key].astype(str) + "_" + pseudoreps.astype(str)
    else:
        # Use original sample IDs if no pseudoreplicates requested
        adata.obs["psbulk_replicate"] = adata.obs[sample_key].astype(str)

    # Call decoupler's pseudobulk function to perform the actual aggregation
    return dc.pp.pseudobulk(
        adata,
        sample_col="psbulk_replicate",  # Column containing our sample/pseudoreplicate IDs
        groups_col=group_key,  # Optional column for separate aggregation (e.g., cell types)
        layer=layer,  # Which data layer to use
        mode=mode,  # How to aggregate (sum, mean, median)
        **kwargs,  # Pass additional parameters to decoupler
    )
