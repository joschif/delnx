"""Dispersion estimation for RNA-seq data analysis.

This module provides functionality to estimate dispersion parameters for negative
binomial models from RNA-seq count data. Accurate dispersion estimation is crucial
for differential expression analysis using methods like DESeq2 or edgeR, especially
for datasets with limited replication.

The implementation supports various estimation methods including:
- DESeq2-style estimation with gamma-distributed trend fitting
- EdgeR-style estimation with log-linear trend shrinkage
- Maximum likelihood estimation (MLE)
- Method of moments estimation

All methods support batch processing for efficient computation with large datasets.
"""

import jax.numpy as jnp
import numpy as np
import tqdm
from anndata import AnnData

from delnx._typing import Method
from delnx._utils import _get_layer, _to_dense
from delnx.models import DispersionEstimator


def _estimate_dispersion_batched(
    X: jnp.ndarray,
    method: str = "deseq2",
    dispersion_range: tuple[float, float] = (1e-6, 10.0),
    shrinkage_weight_range: tuple[float, float] = (0.1, 0.95),
    prior_variance: float = 0.1,
    prior_df: float = 5.0,
    batch_size: int = 2048,
    verbose: bool = True,
) -> jnp.ndarray:
    """Estimate dispersion parameters for negative binomial regression in batches.

    This internal function implements batch processing for dispersion estimation to
    efficiently handle large datasets. It supports multiple methods including DESeq2-style
    and EdgeR-style approaches with trend-based shrinkage.

    Parameters
    ----------
    X : jnp.ndarray
        Expression data matrix, shape (n_cells, n_features). Should contain count data
        that has been normalized by size factors if necessary.
    method : str, default="deseq2"
        Dispersion estimation method:
        - "deseq2": DESeq2-inspired estimation with Bayesian shrinkage towards a
          parametric trend based on a gamma distribution.
        - "edger": EdgeR-inspired estimation with empirical Bayes shrinkage towards
          a log-linear trend.
        - "mle": Maximum likelihood estimation without shrinkage.
        - "moments": Method of moments estimation (faster but less accurate).
    dispersion_range : tuple[float, float], default=(1e-6, 10.0)
        Allowed range for dispersion values, specified as (min_dispersion, max_dispersion).
        Values outside this range will be clipped.
    shrinkage_weight_range : tuple[float, float], default=(0.1, 0.95)
        Range for the shrinkage weight used in DESeq2 and EdgeR methods, specified as
        (min_weight, max_weight). Controls the balance between gene-specific estimates
        and the fitted trend.
    prior_variance : float, default=0.25
        Prior variance parameter for DESeq2-style dispersion shrinkage. Higher values
        result in less shrinkage.
    prior_df : float, default=5.0
        Prior degrees of freedom for edgeR-style dispersion shrinkage. Higher values
        result in stronger shrinkage towards the trend.
    batch_size : int, default=2048
        Number of features to process per batch. Adjust based on available memory.
    verbose : bool, default=True
        Whether to display progress information during computation.

    Returns
    -------
    jnp.ndarray
        Dispersion estimates for each feature, shape (n_features,).
    """
    n_features = X.shape[1]
    estimator = DispersionEstimator(
        dispersion_range=dispersion_range,
        shrinkage_weight_range=shrinkage_weight_range,
        prior_variance=prior_variance,
        prior_df=prior_df,
    )
    estimation_method = "mle" if method in ["mle", "deseq2"] else "moments"

    # Batched estimation of initial dispersion
    init_dispersions = []
    for i in tqdm.tqdm(range(0, n_features, batch_size), disable=not verbose):
        batch = slice(i, min(i + batch_size, n_features))
        X_batch = jnp.asarray(_to_dense(X[:, batch]), dtype=jnp.float32)
        dispersion = estimator.estimate_dispersion(X_batch, method=estimation_method)
        init_dispersions.append(dispersion)

    init_dispersions = jnp.concatenate(init_dispersions, axis=0)

    if method in ["mle", "moments"]:
        # If using MLE or moments, return initial estimates directly
        return init_dispersions

    # Shrinkage of dispersion towards trend
    mean_counts = jnp.array(X.mean(axis=0)).flatten()
    dispersions = estimator.shrink_dispersion(
        dispersions=init_dispersions,
        mu=mean_counts,
        method=method,
    )

    return dispersions


def dispersion(
    adata: AnnData,
    layer: str | None = None,
    size_factor_key: str | None = None,
    method: Method = "deseq2",
    var_key_added: str = "dispersion",
    dispersion_range: tuple[float, float] = (1e-4, 10.0),
    shrinkage_weight_range: tuple[float, float] = (0.1, 0.95),
    prior_variance: float = 0.1,
    prior_df: float = 5.0,
    batch_size: int = 2048,
    verbose: bool = True,
) -> None:
    """Estimate dispersion parameters from (signel-cell) RNA-seq data.

    This function estimates gene-specific dispersion parameters for negative binomial
    models from count data. These dispersion estimates are crucial for differential
    expression analysis with methods like negative binomial regression or DESeq2-style
    approaches. The function supports various estimation methods with trend-based
    shrinkage to improve estimates for genes with low counts or few replicates.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix containing expression data. The data should contain
        raw or normalized counts.
    layer : str | None, default=None
        Layer in `adata.layers` containing count data to use for dispersion estimation.
        If None, uses `adata.X`. Should contain raw counts or normalized counts.
    size_factor_key : str | None, default=None
        Key in `adata.obs` containing size factors for normalization. If provided,
        counts will be normalized by these factors before dispersion estimation.
        This is important for accurate dispersion estimation in datasets with
        variable sequencing depth.
    method : Method, default="deseq2"
        Method for dispersion estimation:
        - "deseq2": DESeq2-inspired estimation with Bayesian shrinkage towards a parametric trend based on a gamma distribution.
        - "edger": EdgeR-inspired estimation with empirical Bayes shrinkage towards a log-linear trend.
        - "mle": Maximum likelihood estimation without shrinkage.
        - "moments": Simple method of moments estimation (faster but less accurate).
    var_key_added : str, default="dispersion"
        Key in `adata.var` where the estimated dispersion values will be stored.
        Existing values will be overwritten.
    dispersion_range : tuple[float, float], default=(1e-4, 10.0)
        Allowed range for dispersion values, specified as (min_dispersion, max_dispersion).
        Values outside this range will be clipped.
    shrinkage_weight_range : tuple[float, float], default=(0.1, 0.95)
        Range for the shrinkage weight used in DESeq2 and EdgeR methods, specified as
        (min_weight, max_weight). Controls how strongly individual gene estimates
        are shrunk towards the trend.
    prior_variance : float, default=0.1
        Prior variance parameter for DESeq2-style dispersion shrinkage. Higher values
        result in less shrinkage.
    prior_df : float, default=5.0
        Prior degrees of freedom for edgeR-style dispersion shrinkage. Higher values
        result in stronger shrinkage towards the trend.
    batch_size : int, default=2048
        Number of features to process per batch. Adjust based on available memory
        and dataset size.
    verbose : bool, default=True
        Whether to display progress information during computation.

    Returns
    -------
    Updates ``adata`` in place and sets the following fields:

        - ``adata.var[var_key_added]``: Estimated dispersion values for each feature.

    Examples
    --------
    Estimate dispersions using the DESeq2 method:

    >>> import scanpy as sc
    >>> import delnx as dx
    >>> adata = sc.read_h5ad("counts.h5ad")
    >>> # Calculate size factors first (optional but recommended)
    >>> adata.obs["size_factors"] = adata.X.sum(axis=1) / np.median(adata.X.sum(axis=1))
    >>> # Estimate dispersions
    >>> dx.pp.dispersion(adata, size_factor_key="size_factors", method="deseq2")

    Using different estimation methods:

    >>> # EdgeR-style estimation
    >>> dx.pp.dispersion(adata, size_factor_key="size_factors", method="edger", var_key_added="disp_edger")
    >>> # Maximum likelihood estimation (no shrinkage)
    >>> dx.pp.dispersion(adata, size_factor_key="size_factors", method="mle", var_key_added="disp_mle")

    Notes
    -----
    - For accurate dispersion estimation, it's recommended to use raw counts and
      provide size factors for normalization.
    - For very large datasets, consider increasing the batch size if memory allows,
      or decreasing it for memory-constrained environments.
    - The estimated dispersions can be used for differential expression analysis
      with the negative binomial model by providing the `var_key_added` value
      as the `dispersion_key` parameter in the `de` function.
    """
    # Get expression data from the specified layer or X
    X = _get_layer(adata, layer)

    # Apply size factor normalization if provided
    if size_factor_key is not None:
        if size_factor_key not in adata.obs:
            raise ValueError(f"Size factor key '{size_factor_key}' not found in adata.obs")
        size_factors = adata.obs[size_factor_key].values
        X_norm = X / size_factors[:, None]
    else:
        X_norm = X

    # Estimate dispersions using the specified method
    dispersions = _estimate_dispersion_batched(
        X=X_norm,
        method=method,
        dispersion_range=dispersion_range,
        shrinkage_weight_range=shrinkage_weight_range,
        prior_variance=prior_variance,
        prior_df=prior_df,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Store results in adata.var
    adata.var[var_key_added] = np.array(dispersions)
