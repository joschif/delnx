"""Size factor computation for RNA-seq normalization.

This module provides methods to compute size factors that account for differences
in sequencing depth and technical biases between samples in RNA-seq data. Size factors
are crucial for accurate differential expression analysis and can be used as offset terms
in count-based regression models.

The module implements several normalization methods:
- DESeq2-style median-of-ratios size factors
- Quantile regression-based normalization (similar to SCnorm)
- Library size normalization (sequencing depth)

These size factors can be used for:
1. Scaling raw counts for visualization
2. Providing offset terms for negative binomial regression models
3. Normalizing counts prior to log-transformation
"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import sparse

from delnx._utils import _get_layer, _to_dense
from delnx.models import LinearRegression


def _compute_library_size(adata, layer=None):
    """Compute library size factors for each cell.

    This function calculates size factors based on the total count (library size)
    of each cell, normalized by the mean library size across all cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data.
    layer : str, optional
        Layer in `adata.layers` to use for calculation. If `None`, uses `adata.X`.

    Returns
    -------
    numpy.ndarray
        Array of size factors with the same length as the number of observations in `adata`.
    """
    # Get expression matrix
    X = _get_layer(adata, layer)

    if sparse.issparse(X):
        libsize = np.asarray(X.sum(axis=1)).flatten()
    else:
        libsize = X.sum(axis=1)

    return libsize / np.mean(libsize)


def _compute_median_ratio(adata, layer=None):
    """Compute DESeq2-style median-of-ratios size factors.

    This function implements the DESeq2 normalization method which computes size factors
    as the median of ratios of gene expression to a reference sample (geometric mean
    across all samples). This approach is robust to differential expression between
    samples and is recommended for bulk RNA-seq and single-cell data with sufficient
    coverage.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data.
    layer : str, optional
        Layer in `adata.layers` to use for calculation. If None, uses `adata.X`.

    Returns
    -------
    numpy.ndarray
        Array of size factors with the same length as the number of observations in `adata`.

    Raises
    ------
    ValueError
        If all genes have zero counts across all samples.
    """
    X = _get_layer(adata, layer)

    # Compute gene-wise mean log counts
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_X = np.log(X)

    log_means = log_X.mean(0)

    # Filter out genes with -∞ log means (genes with all zero counts)
    filtered_genes = ~np.isinf(log_means)

    # Check if we have any genes left after filtering
    if not filtered_genes.any():
        raise ValueError("All genes have all-zero counts. Cannot compute size factors with median-of-ratios method.")

    # Compute log ratios using only filtered genes
    log_ratios = log_X[:, filtered_genes] - log_means[filtered_genes]

    # Compute sample-wise median of log ratios
    log_medians = np.median(log_ratios, axis=1)
    size_factors = np.exp(log_medians)

    return size_factors


@partial(jax.jit, static_argnums=(2,))
def _fit_lm(x, y, maxiter=100):
    """Fit a linear model using JAX.

    Parameters
    ----------
    x : jax.numpy.ndarray
        Input features, shape (n_samples, n_features).
    y : jax.numpy.ndarray
        Target values, shape (n_samples,).
    maxiter : int, default=100
        Maximum number of iterations for optimization.

    Returns
    -------
    jax.numpy.ndarray
        Predicted values from the linear model.
    """
    model = LinearRegression(skip_stats=True, maxiter=maxiter)
    results = model.fit(x, y)
    pred = x @ results["coef"]
    return pred


_fit_lm_batch = jax.vmap(_fit_lm, in_axes=(None, 1), out_axes=0)  # Vectorized version for batch processing


def _compute_quantile_regression(adata, layer=None, min_counts=1, quantiles=np.linspace(0.1, 0.9, 9), batch_size=32):
    """Compute size factors using quantile regression.

    This function implements a quantile regression-based approach similar to SCnorm,
    which accounts for gene-specific count-depth relationships. It is particularly
    useful for single-cell RNA-seq data where library size normalization may not
    adequately correct for technical biases.

    The method:
    1. Groups genes into quantile bins based on their mean expression
    2. For each bin, fits linear models to predict gene expression from cell-specific medians
    3. Combines predictions across bins to compute size factors

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data.
    layer : str, optional
        Layer in `adata.layers` to use for calculation. If None, uses `adata.X`.
    min_counts : float, default=1
        Minimum mean count threshold for genes to be included in normalization.
    quantiles : numpy.ndarray, default=np.linspace(0.1, 0.9, 9)
        Quantile points for binning genes based on mean expression.
    batch_size : int, default=32
        Number of genes to process per batch for memory efficiency.

    Returns
    -------
    numpy.ndarray
        Array of size factors with the same length as the number of observations in `adata`.
    """
    # Get count matrix and filter genes
    X = _get_layer(adata, layer)  # shape: (cells x genes)
    gene_means = np.asarray(X.mean(axis=0)).flatten()  # per-gene means
    valid_genes = gene_means >= min_counts
    counts = X[:, valid_genes]
    gene_means = gene_means[valid_genes]

    # Log-transform
    log_counts = np.log1p(counts)
    quantile_bins = pd.qcut(gene_means, q=quantiles, labels=False, duplicates="drop")

    n_cells = log_counts.shape[0]
    size_factor_numerators = np.zeros(n_cells)
    total_weight = 0

    for bin_idx in np.unique(quantile_bins):
        group_idx = np.where(quantile_bins == bin_idx)[0]
        if len(group_idx) < 10:
            continue

        # Median expression per cell across genes in the group
        median_expr = np.median(log_counts[:, group_idx], axis=1).reshape(-1, 1)  # shape: (n_cells, 1)

        for i in range(0, len(group_idx), batch_size):
            batch = slice(i, min(i + batch_size, len(group_idx)))
            y = jnp.array(_to_dense(log_counts[:, group_idx[batch]]))  # shape: (n_cells, batch_size)
            preds = _fit_lm_batch(median_expr, y)  # shape: (batch_size, n_cells)
            size_factor_numerators += preds.sum(axis=0)
            total_weight += preds.shape[0]

    size_factors = size_factor_numerators / total_weight
    return size_factors / np.mean(size_factors)


def size_factors(adata, method="ratio", layer=None, obs_key_added="size_factor", **kwargs):
    """Compute size factors for (single-cell) RNA-seq normalization.

    This function calculates sample/cell-specific normalization factors (size factors)
    to account for differences in sequencing depth and technical biases between samples.
    The computed size factors can be used to normalize counts for visualization or
    as offset terms in statistical models for differential expression analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data.
    method : str, default="ratio"
        Method to compute size factors:
            - "ratio": DESeq2-style median-of-ratios size factors, robust to differential expression between samples. Recommended for bulk RNA-seq and well-covered single-cell data.
            - "quantile_regression": SCnorm-style quantile regression normalization, accounts for gene-specific count-depth relationships. Recommended for single-cell RNA-seq data with potential gene-dependent biases.
            - "library_size": Library size normalization based on the total counts per sample. Simple but less robust to highly expressed genes or differential expression.
    layer : str, optional
        Layer in `adata.layers` to use for size factor calculation. If None, uses `adata.X`. Should contain raw (unlogged) counts.
    obs_key_added : str, default="size_factor"
        Key in `adata.obs` where the computed size factors will be stored.
    **kwargs : dict
        Additional parameters for specific methods:
            - For "quantile_regression": `min_counts` (default=1), `quantiles`, `batch_size` (default=32)

    Returns
    -------
    Updates ``adata`` in place and sets the following fields:

            - `adata.obs[obs_key_added]`: Size factors for each cell.

    Examples
    --------
    Calculate DESeq2-style median-of-ratios size factors:

    >>> import scanpy as sc
    >>> import delnx as dx
    >>> adata = sc.read_h5ad("counts.h5ad")
    >>> dx.pp.size_factors(adata, method="ratio", obs_key_added="size_factors")

    Use different normalization methods:

    >>> # Library size normalization
    >>> dx.pp.size_factors(adata, method="library_size", obs_key_added="lib_size_factors")
    >>> # Quantile regression normalization (SCnorm-style)
    >>> dx.pp.size_factors(adata, method="quantile_regression", obs_key_added="qr_factors", min_counts=5)

    Use size factors for normalization in differential expression analysis:

    >>> # Compute DE with size factors as offset
    >>> results = dx.tl.de(adata, condition_key="treatment", size_factor_key="size_factors")

    Notes
    -----
    - Size factors are scaled to have a mean of 1.0 across all samples
    - A warning will be raised if any size factors are zero or negative
    """
    if method == "ratio":
        size_factors = _compute_median_ratio(adata, layer)
    elif method == "quantile_regression":
        size_factors = _compute_quantile_regression(adata, layer=layer, **kwargs)
    elif method == "library_size":
        size_factors = _compute_library_size(adata, layer)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Warn if size factors contain zeros
    if np.any(size_factors <= 0):
        warnings.warn(
            "Size factors contain zero or negative values. This may indicate issues with the data and can be problematic for downstream analyses.",
            stacklevel=2,
        )

    # Store size factors in adata.obs
    adata.obs[obs_key_added] = size_factors
