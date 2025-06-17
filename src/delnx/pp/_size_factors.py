"""Size factor computation for (single-cell) RNA-seq data."""

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
    """Compute library size factors for each cell."""
    # Get expression matrix
    X = _get_layer(adata, layer)

    if sparse.issparse(X):
        libsize = np.asarray(X.sum(axis=1)).flatten()
    else:
        libsize = X.sum(axis=1)

    return libsize / np.mean(libsize)


def _compute_median_ratio(adata, layer=None):
    """Compute median-of-ratios size factors."""
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
    model = LinearRegression(skip_wald=True, maxiter=maxiter)
    results = model.fit(x, y)
    pred = x @ results["coef"]
    return pred


_fit_lm_batch = jax.vmap(_fit_lm, in_axes=(None, 1), out_axes=0)


def _compute_quantile_regression(adata, layer=None, min_counts=1, quantiles=np.linspace(0.1, 0.9, 9), batch_size=32):
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


def size_factors(adata, method="library_size", layer=None, **kwargs):
    """Compute size factors for normalization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    method : str, optional
        Method to compute size factors. Options are:
        - "ratio": DESeq2-style median-of-ratios size factor
        - "quantile_regression": SCnorm-style quantile regression normalization
        - "library_size": Library size normalization (sum of counts)
    layer : str, optional
        Layer to use for size factor calculation. If None, use adata.X.
    **kwargs : dict
        Additional parameters for specific methods.

    Returns
    -------
    None
        Size factors are stored in adata.obs.
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
    adata.obs["size_factor"] = size_factors
