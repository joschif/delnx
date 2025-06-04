"""Effect size calculation and evaluation functions for differential expression analysis."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tqdm
from anndata import AnnData

from delnx._typing import ComparisonMode, DataType
from delnx._utils import _get_layer, _to_dense

from ._utils import _infer_data_type, _validate_conditions


def _log2fc(
    X: np.ndarray,
    condition_mask: np.ndarray,
    data_type: DataType,
    eps: float = 1e-8,
) -> np.ndarray:
    """Calculate log2 fold changes between two conditions.

    Parameters
    ----------
    X
        Expression matrix
    condition_mask
        Boolean mask indicating reference condition (True) vs test condition (False)
    data_type
        Type of data:
        - counts: Raw count data
        - lognorm: Log-normalized data (assumed to be log1p of normalized counts)
        - binary: Binary data
    eps
        Small constant to add for numerical stability with raw counts

    Returns
    -------
    np.ndarray
        Log2 fold changes
    """
    # Extract test and reference data once
    ref_data = X[~condition_mask]
    test_data = X[condition_mask]

    if data_type == "lognorm":
        # For log-normalized data (log1p transformed):
        # More efficient approach: directly transform to linear space,
        # compute means, then take log2 ratio
        ref_data = np.expm1(ref_data.astype(np.float64))
        test_data = np.expm1(test_data.astype(np.float64))
    elif data_type in ["counts", "binary"]:
        # For raw counts or binary, calculate ratio of means directly
        pass
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    ref_means = ref_data.mean(axis=0) + eps
    test_means = test_data.mean(axis=0) + eps
    log2fc = np.log2(test_means / ref_means)
    return np.asarray(log2fc, dtype=np.float64).flatten()


def log2fc(
    adata: AnnData,
    condition_key: str,
    reference: str | tuple[str, str] | None = None,
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    data_type: DataType = "auto",
    min_samples: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calculate log2 fold changes between condition levels.

    Parameters
    ----------
    adata
        AnnData object
    condition_key
        Column in adata.obs containing condition values
    reference
        Reference condition level or tuple (reference, comparison)
    mode
        How to perform comparisons. One of:
        - all_vs_ref: Compare all levels to reference
        - all_vs_all: Compare all pairs of levels
        - 1_vs_1: Compare only two levels (reference and comparison group)
    layer
        Layer in adata to use
    data_type
        Type of data:
        - counts: Raw count data
        - lognorm: Log-normalized data (assumed to be log1p of normalized counts)
        - binary: Binary data
        - auto: Automatically infer data type
    min_samples
        Minimum number of samples per condition level
    verbose
        Whether to print progress messages

    Returns
    -------
    pandas.DataFrame
        Log2 fold change results with columns:
        - feature: Gene or feature ID
        - test_condition: Test condition name
        - ref_condition: Reference condition name
        - log2fc: Log2 fold change value
        - group: Cell group (if group_key is specified)
    """
    # Validate inputs
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    # Get condition values
    condition_values = adata.obs[condition_key].values
    levels, comparisons = _validate_conditions(condition_values, reference, mode)

    # Get expression matrix
    X = _get_layer(adata, layer)

    # Infer data type if auto
    if data_type == "auto":
        data_type = _infer_data_type(X)
        if verbose:
            print(f"Inferred data type: {data_type}")
    elif verbose:
        print(f"Using specified data type: {data_type}")

    # Calculate log2fc for each comparison
    results = []
    for group1, group2 in comparisons:
        # Get cell masks
        mask1 = adata.obs[condition_key].values == group1
        mask2 = adata.obs[condition_key].values == group2

        if np.sum(mask1) < min_samples or np.sum(mask2) < min_samples:
            if verbose:
                print(f"Skipping comparison {group1} vs {group2} with < {min_samples} samples")
            continue

        all_mask = mask1 | mask2

        # Get data for calculations
        data = X[all_mask, :]
        condition_mask = adata.obs.loc[all_mask, condition_key].values == group1

        # Calculate log2 fold change
        log2fc_values = _log2fc(
            X=data,
            condition_mask=condition_mask,
            data_type=data_type,
        )

        # Create results dataframe
        result_df = pd.DataFrame(
            {
                "feature": adata.var_names,
                "test_condition": group1,
                "ref_condition": group2,
                "log2fc": log2fc_values,
            }
        )

        results.append(result_df)

    if len(results) == 0:
        raise ValueError("No valid comparisons found for fold change analysis")

    # Combine results
    return pd.concat(results, axis=0)


@jax.jit
def _auroc(x: jnp.ndarray, groups: jnp.ndarray) -> float:
    """Calculate AUROC for a single feature.

    Parameters
    ----------
    x
        Feature values for all cells
    groups
        Binary array (1 for positive class, 0 for negative class)

    Returns
    -------
    float
        Area under the ROC curve
    """
    # Sort scores and corresponding truth values (highest scores first)
    desc_score_indices = jnp.argsort(x)[::-1]
    x = x[desc_score_indices]
    groups = groups[desc_score_indices]

    # x typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = jnp.array(jnp.diff(x) != 0, dtype=jnp.int32)
    threshold_mask = jnp.r_[distinct_value_indices, 1]

    # Accumulate the true positives with decreasing threshold
    tps_ = jnp.cumsum(groups)  # True positives
    fps_ = 1 + jnp.arange(groups.size) - tps_  # False positives

    # Mask out the values that are not distinct
    tps = jnp.sort(tps_ * threshold_mask)
    fps = jnp.sort(fps_ * threshold_mask)

    # Prepend 0 to start the curve at the origin
    tps = jnp.r_[0, tps]
    fps = jnp.r_[0, fps]

    # Calculate TPR and FPR
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # Calculate area using trapezoidal rule
    area = jnp.trapezoid(tpr, fpr)
    return area


_auroc_batch = jax.vmap(_auroc, in_axes=[1, None])


def _batched_auroc(
    X: np.ndarray,
    groups: np.ndarray,
    batch_size: int = 1024,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run AUROC analysis in batches.

    Parameters
    ----------
    X
        Expression data matrix, shape (n_cells, n_features)
    groups
        Group labels (1 for positive class, 0 for negative class), shape (n_cells,)
    batch_size
        Number of features to process per batch
    verbose
        Whether to show progress bar

    Returns
    -------
    pandas.DataFrame
        DataFrame with AUROC results for each feature
    """
    # Process in batches
    n_features = X.shape[1]

    # Convert groups to JAX array
    groups_jax = jnp.array(groups, dtype=jnp.int32)

    # Process all batches
    results = []
    for i in tqdm.tqdm(range(0, n_features, batch_size), disable=not verbose):
        batch = slice(i, min(i + batch_size, n_features))
        X_batch = jnp.asarray(_to_dense(X[:, batch]), dtype=jnp.float32)

        # Calculate AUROC values for batch using vectorized function
        auroc_values = _auroc_batch(X_batch, groups_jax)

        results.append(auroc_values)

    # Concatenate results
    results = np.concatenate(results, axis=0)
    return results


def auroc(
    adata: AnnData,
    condition_key: str,
    reference: str | tuple[str, str] | None = None,
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    min_samples: int = 2,
    batch_size: int = 1024,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate Area Under the Receiver Operating Characteristic (AUROC) between condition levels.

    Parameters
    ----------
    adata
        AnnData object
    condition_key
        Column in adata.obs containing condition values
    reference
        Reference condition level or tuple (reference, comparison)
    mode
        How to perform comparisons. One of:
        - all_vs_ref: Compare all levels to reference
        - all_vs_all: Compare all pairs of levels
        - 1_vs_1: Compare only two levels (reference and comparison group)
    layer
        Layer in adata to use
    min_samples
        Minimum number of samples per condition level
    batch_size
        Size of feature batches for processing to avoid memory issues
    verbose
        Whether to print progress messages

    Returns
    -------
    pandas.DataFrame
        AUROC results with columns:
        - feature: Gene or feature ID
        - test_condition: Test condition name
        - ref_condition: Reference condition name
        - auroc: AUROC value
        - group: Cell group (if group_key is specified)
    """
    # Validate inputs
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")

    # Get condition values
    condition_values = adata.obs[condition_key].values
    _, comparisons = _validate_conditions(condition_values, reference, mode)

    # Get expression matrix
    X = _get_layer(adata, layer)

    # Calculate AUROC for each comparison
    results = []
    for group1, group2 in comparisons:
        # Get cell masks
        mask1 = adata.obs[condition_key].values == group1
        mask2 = adata.obs[condition_key].values == group2

        if np.sum(mask1) < min_samples or np.sum(mask2) < min_samples:
            if verbose:
                print(f"Skipping comparison {group1} vs {group2} with < {min_samples} samples")
            continue

        all_mask = mask1 | mask2

        # Get data for calculations
        data = X[all_mask, :]

        # Create binary groups vector (1 for test group, 0 for reference group)
        groups = (adata.obs.loc[all_mask, condition_key].values == group1).astype(np.int32)

        # Run batched AUROC calculation
        auroc_values = _batched_auroc(
            X=data,
            groups=groups,
            batch_size=batch_size,
            verbose=verbose,
        )

        # Create results dataframe
        result_df = pd.DataFrame(
            {
                "feature": adata.var_names,
                "test_condition": group1,
                "ref_condition": group2,
                "auroc": auroc_values,
            }
        )

        results.append(result_df)

    if len(results) == 0:
        raise ValueError("No valid comparisons found for AUROC analysis")

    # Combine results
    return pd.concat(results, axis=0)
