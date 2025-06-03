from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm
from orchard._utils import _get_layer, _to_dense
from orchard.models import LinearRegression
from scipy import sparse
from scipy.stats import gmean


def _compute_library_size(adata, layer=None):
    """Compute library size factors for each cell."""
    # Get expression matrix
    X = _get_layer(adata, layer)

    if sparse.issparse(X):
        libsize = np.asarray(X.sum(axis=1)).flatten()
    else:
        libsize = X.sum(axis=1)

    size_factors = libsize / np.mean(libsize)
    adata.obs["size_factor_libsize"] = size_factors


def _compute_median_ratio(adata, layer=None):
    """Compute DESeq2-style median ratio size factors."""
    # Only really works for (dense) pseudobulk data
    X = _get_layer(adata, layer)

    if sparse.issparse(X):
        raise ValueError("Method 'median_ratio' requires a dense matrix.")

    geometric_means = gmean(X + 1e-6, axis=0)
    ratios = X / geometric_means
    size_factors = np.median(ratios, axis=1)
    adata.obs["size_factor_mratio"] = size_factors / np.mean(size_factors)


def _masked_quantile(x, mask, q):
    """Compute quantile from masked array using jax operations."""
    x_masked = jnp.where(mask, x, jnp.inf)
    x_sorted = jnp.sort(x_masked)
    n_valid = jnp.sum(mask)
    idx = jnp.clip(jnp.floor(q * (n_valid - 1)).astype(int), 0, x.shape[0] - 1)
    return x_sorted[idx]


@partial(jax.jit, static_argnums=(2, 3))
def _tmm(X, ref_data, logratio_trim, abs_expr_trim):
    """Compute TMM factor for a single column using JAX.

    Parameters
    ----------
    X : jnp.ndarray
        Observed expression data for a single sample (column)
    ref_data : tuple
        Tuple containing (reference_expression, reference_libsize)
    logratio_trim : float
        Proportion of log fold changes to trim
    abs_expr_trim : float
        Proportion of average expression values to trim

    Returns
    -------
    factor : float
        TMM normalization factor
    """
    ref, libsize_ref = ref_data
    libsize_X = jnp.sum(X)
    eps = 1e-6  # Small constant for numerical stability

    # Calculate log ratio and absolute expression
    logR = jnp.log2((X / libsize_X + eps) / (ref / libsize_ref + eps))
    absE = 0.5 * (jnp.log2(X / libsize_X + eps) + jnp.log2(ref / libsize_ref + eps))

    # Calculate variance
    v = (1.0 / jnp.clip(X, 1e-3)) + (1.0 / jnp.clip(ref, 1e-3))

    # Filter for finite values
    finite_mask = jnp.isfinite(logR) & jnp.isfinite(absE)
    logR = jnp.where(finite_mask, logR, 0.0)
    absE = jnp.where(finite_mask, absE, 0.0)
    v = jnp.where(finite_mask, v, jnp.inf)

    # Get quantiles for trimming
    lo_lrt = _masked_quantile(logR, finite_mask, logratio_trim)
    hi_lrt = _masked_quantile(logR, finite_mask, 1 - logratio_trim)
    lo_aet = _masked_quantile(absE, finite_mask, abs_expr_trim)
    hi_aet = _masked_quantile(absE, finite_mask, 1 - abs_expr_trim)

    # Apply trimming
    keep = (logR > lo_lrt) & (logR < hi_lrt) & (absE > lo_aet) & (absE < hi_aet) & finite_mask

    # Calculate weights and weighted log fold change
    w = 1.0 / v
    w = jnp.where(keep, w, 0.0)
    numer = jnp.sum(w * logR)
    denom = jnp.sum(w)

    # Calculate TMM factor
    factor = jnp.where(denom > 0, 2.0 ** (numer / denom), 1.0)
    return factor


_tmm_batch = jax.vmap(_tmm, in_axes=(0, None, None, None), out_axes=0)


def _compute_TMM(adata, layer=None, ref_col=None, logratio_trim=0.3, abs_expr_trim=0.05, batch_size=64, verbose=False):
    """Compute edgeR-style TMM normalization size factors using JAX.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data
    layer : str, optional
        Layer of expression data to use. If None, use adata.X
    logratio_trim : float, default=0.3
        Proportion of log fold changes to trim (0-0.5)
    abs_expr_trim : float, default=0.05
        Proportion of average expression values to trim (0-0.5)
    """
    # Get expression matrix
    X = _get_layer(adata, layer)
    n_cells = X.shape[0]
    ref_obs = np.argmin(np.abs(np.sum(X, axis=1) - np.median(np.sum(X, axis=1))))

    # Get reference data
    ref = X[ref_obs, :]
    libsize_ref = np.sum(ref)
    ref_data = (ref, libsize_ref)

    results = []
    for i in tqdm.tqdm(range(0, n_cells, batch_size), disable=not verbose):
        # Get current batch of features
        batch = slice(i, min(i + batch_size, n_cells))
        X_batch = jnp.asarray(_to_dense(X[batch, :]), dtype=jnp.float32)
        tmm_factors = _tmm_batch(X_batch, ref_data, logratio_trim, abs_expr_trim)
        results.append(tmm_factors)

    # Concatenate results
    tmm_factors = np.concatenate(results)
    adata.obs["size_factor_TMM"] = tmm_factors / np.mean(tmm_factors)


@partial(jax.jit, static_argnums=(2,))
def _fit_lm(x, y, maxiter=100):
    model = LinearRegression(skip_wald=True, maxiter=maxiter)
    results = model.fit(x, y)
    pred = x @ results["coef"]
    return pred


_fit_lm_batch = jax.vmap(_fit_lm, in_axes=(None, 1), out_axes=0)


def _compute_quantile_regression(
    adata, layer=None, min_counts=1, quantiles=np.linspace(0.1, 0.9, 9), batch_size=32, verbose=False
):
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

        for i in tqdm.trange(0, len(group_idx), batch_size, disable=not verbose):
            batch = slice(i, min(i + batch_size, len(group_idx)))
            y = jnp.array(_to_dense(log_counts[:, group_idx[batch]]))  # shape: (n_cells, batch_size)
            preds = _fit_lm_batch(median_expr, y)  # shape: (batch_size, n_cells)
            size_factor_numerators += preds.sum(axis=0)
            total_weight += preds.shape[0]

    size_factors = size_factor_numerators / total_weight
    size_factors /= np.mean(size_factors)

    adata.obs["size_factor_qreg"] = size_factors


def size_factors(adata, method="median_ratio", layer=None, **kwargs):
    """Compute size factors for normalization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    method : str, optional
        Method to compute size factors. Options are:
        - "median_ratio": DESeq2-style median ratio
        - "TMM": edgeR-style TMM normalization
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
    if method == "median_ratio":
        _compute_median_ratio(adata, layer)
    elif method == "quantile_regression":
        _compute_quantile_regression(adata, layer=layer, **kwargs)
    elif method == "TMM":
        _compute_TMM(adata, layer=layer, **kwargs)
    elif method == "library_size":
        _compute_library_size(adata, layer)
    else:
        raise ValueError(f"Unsupported method: {method}")
