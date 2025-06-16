"""Estimate dispersion fromm (single-cell) RNA-seq data."""

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
    dispersion_range: tuple[float, float] = (1e-4, 10.0),
    shrinkage_weight_range: tuple[float, float] = (0.05, 0.95),
    prior_variance: float = 0.25,
    prior_df: float = 10.0,
    batch_size: int = 2048,
    verbose: bool = True,
) -> jnp.ndarray:
    """Estimate dispersion for negative binomial regression.

    Parameters
    ----------
        X: Expression data matrix, shape (n_cells, n_features)
        method: Dispersion estimation method:
            - "deseq2": DESeq2-inspired dispersion estimation with bayesian shrinkage towards a parametric trend based on a gamma distribution.
            - "edger": EdgeR-inspired dispersion estimation with empirical Bayes shrinkage towards a log-linear trend.
            - "mle": Maximum likelihood estimation of dispersion.
            - "moments": Simple method of moments dispersion estimation.
        dispersion_range: Allowed range for dispersion values.
        shrinkage_weight_range: Range for the shrinkage weight used in DESeq2 and EdgeR methods.
        prior_variance: Prior variance for DESeq2-style dispersion shrinkage.
        prior_df: Prior degrees of freedom for edgeR-style dispersion shrinkage.
        batch_size: Number of features to process per batch.


    Returns
    -------
        Dispersion estimates for each feature
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
    shrinkage_weight_range: tuple[float, float] = (0.05, 0.95),
    prior_variance: float = 0.25,
    prior_df: float = 10.0,
    batch_size: int = 2048,
    verbose: bool = True,
) -> DispersionEstimator:
    """Estimate dispersion from (single-cell) RNA-seq data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data.
    layer : str, optional
        Layer of the AnnData object with counts to use for dispersion estimation. If None, uses `adata.X`.
    size_factor_key : str, optional
        Key in `adata.obs` containing size factors for normalization. If provided, they will be used for normalization before dispersion estimation.
    method : str, optional
        Method for dispersion estimation. Options are:
        - "deseq2": DESeq2-inspired dispersion estimation with bayesian shrinkage towards a parametric trend based on a gamma distribution.
        - "edger": EdgeR-inspired dispersion estimation with empirical Bayes shrinkage towards a log-linear trend.
        - "mle": Maximum likelihood estimation of dispersion.
        - "moments": Simple method of moments dispersion estimation.
    var_key_added : str, optional
        Key in `adata.var` where the estimated dispersion values will be stored.
    dispersion_range : tuple, optional
        Range of dispersion values to consider.
    shrinkage_weight_range : tuple, optional
        Range of shrinkage weights used in DESeq2 and EdgeR methods.
    prior_variance : float, optional
        Prior variance for DESeq2-style dispersion shrinkage.
    prior_df : float, optional
        Prior degrees of freedom for edgeR-style dispersion shrinkage.
    batch_size : int, optional
        Number of features to process in each batch for dispersion estimation.
    verbose : bool, optional
        If True, show progress bars and messages during estimation.

    Returns
    -------
    Returns `None` and sets `adata.var[var_key_added]` with estimated dispersion values.
    """
    X = _get_layer(adata, layer)

    if size_factor_key is not None:
        size_factors = adata.obs[size_factor_key].values
        X /= size_factors[:, None]

    dispersions = _estimate_dispersion_batched(
        X=X,
        method=method,
        dispersion_range=dispersion_range,
        shrinkage_weight_range=shrinkage_weight_range,
        prior_variance=prior_variance,
        prior_df=prior_df,
        batch_size=batch_size,
        verbose=verbose,
    )

    adata.var[var_key_added] = np.array(dispersions)
