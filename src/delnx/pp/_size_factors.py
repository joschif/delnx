import numpy as np
from scipy import sparse

from delnx._logging import logger
from delnx._utils import _get_layer


def size_factors(adata, method="normed_sum", layer=None, obs_key_added="size_factors"):
    """Compute size factors for (single-cell) RNA-seq normalization.

    This function calculates sample/cell-specific normalization factors (size factors)
    to account for differences in sequencing depth and technical biases between samples.
    The computed size factors can be used to normalize counts for visualization or
    as offset terms in statistical models for differential expression analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing expression data.
    method : str, default="normed_sum"
        Method to compute size factors:
        - "normed_sum": Library size normalization based on the total counts per
          sample. Simple and efficient, works with sparse matrices.
        - "ratio": DESeq2-style median-of-ratios size factors, robust to differential
          expression between samples. Requires dense matrices.
        - "poscounts": Positive counts method with geometric mean normalization.
          Requires dense matrices.
    layer : str, optional
        Layer in `adata.layers` to use for size factor calculation. If None, uses
        `adata.X`. Should contain raw (unlogged) counts.
    obs_key_added : str, default="size_factors"
        Key in `adata.obs` where the computed size factors will be stored.

    Returns
    -------
    None
        Updates `adata` in place and sets the following field:
        - `adata.obs[obs_key_added]`: Size factors for each cell.

    Examples
    --------
    Calculate library size normalization (default):

    >>> import scanpy as sc
    >>> import delnx as dx
    >>> adata = sc.read_h5ad("counts.h5ad")
    >>> dx.pp.size_factors(adata, method="normed_sum")

    Calculate DESeq2-style median-of-ratios size factors:

    >>> # Requires dense matrix
    >>> if sparse.issparse(adata.X):
    ...     adata.X = adata.X.toarray()
    >>> dx.pp.size_factors(adata, method="ratio", obs_key_added="ratio_factors")

    Use size factors for normalization in differential expression analysis:

    >>> # Compute DE with size factors as offset
    >>> results = dx.tl.de(adata, condition_key="treatment", size_factor_key="size_factors")

    Notes
    -----
    - Size factors are scaled to have a geometric mean of 1.0 across all samples
    - Methods "ratio" and "poscounts" require dense matrices; use "normed_sum" for sparse data
    - A warning will be raised if size factors cannot be computed for some cells
    - Zero or invalid size factors are replaced with a small value (0.001)
    """
    # Get expression matrix
    X = _get_layer(adata, layer)

    if sparse.issparse(X) and method in ["ratio", "poscounts"]:
        raise ValueError(
            f"The '{method}' method requires a dense matrix. Please convert the sparse matrix to dense format before using this method or use the 'normed_sum' method."
        )

    if method == "ratio":
        log_X = np.log(X)
        log_geo_means = np.mean(log_X, 0)
        filtered_genes = ~np.isinf(log_geo_means)
        # Check if we have any genes left after filtering
        if not filtered_genes.any():
            raise ValueError(
                f"All genes have a least one zero count. Cannot compute size factors with the '{method}' method."
            )
        log_ratios = log_X[:, filtered_genes] - log_geo_means[filtered_genes]
        size_factors = np.exp(np.median(log_ratios, axis=1))

    elif method == "normed_sum":
        if sparse.issparse(X):
            size_factors = np.asarray(X.sum(axis=1)).flatten().astype(float)
        else:
            size_factors = X.sum(axis=1).astype(float)

    elif method == "poscounts":
        log_geometric_means = np.mean(np.log(X + 0.5), axis=0)
        X[X == 0] = np.nan
        size_factors = np.exp(np.nanmedian(np.log(X) / log_geometric_means.reshape(1, -1), axis=0))

    else:
        raise ValueError(f"Unsupported method: {method}")

    zero_col = size_factors <= 0 | np.isnan(size_factors)

    if zero_col.any() and method == "ratio":
        logger.warning(
            "Too many zeros to compute size factors with the 'ratio' method. Please use the 'normed_sum' method instead."
        )

    if zero_col.any():
        logger.warning("Some size factors could not be computed due to too many zero values. Setting to 0.001")
        size_factors[zero_col] = np.nan
        size_factors /= np.exp(np.nanmean(np.log(size_factors)))
        size_factors[zero_col] = 0.001

    else:
        size_factors /= np.exp(np.mean(np.log(size_factors)))

    # Store size factors in adata.obs
    adata.obs[obs_key_added] = size_factors
