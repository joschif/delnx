"""Differential expression testing module."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData

from delnx._typing import ComparisonMode, DataType, Method
from delnx._utils import _get_layer

from ._de_tests import _run_de, _run_deseq2
from ._effects import _batched_auroc, _log2fc
from ._jax_tests import _run_batched_de
from ._utils import (
    _infer_data_type,
    _prepare_model_data,
    _validate_conditions,
)


def de(
    adata: AnnData,
    condition_key: str,
    group_key: str | None = None,
    reference: str | tuple[str, str] | None = None,
    method: Method = "deseq2",
    backend: str = "statsmodels",
    covariates: list[str] | None = None,
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    data_type: DataType = "auto",
    log2fc_threshold: float = 0.2,
    min_samples: int = 2,
    multitest_method: str = "fdr_bh",
    n_jobs: int = 1,
    batch_size: int = 32,
    optimizer: str = "BFGS",
    maxiter: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run differential expression between condition levels.

    Parameters
    ----------
    adata
        AnnData object
    condition_key
        Column in adata.obs containing condition values
    reference
        Reference condition level or tuple (reference, comparison)
    group_key
        Column in adata.obs for grouped DE testing
    method
        Testing method to use. One of:
        - deseq2: DESeq2 for count data
        - negbinom: Negative binomial GLM and wald test
        - lr: Logistic regression and likelihood ratio test
        - anova: ANOVA based on a linear model
        - anova_residual: Linear model with residual F-test
        - binomial: Binomial GLM
    backend
        Backend to use for linear mdoel-based DE methods.
        - jax: Use custom linear models in JAX for batched and GPU-accelerated methods
        - statsmodels: Use linear models from statsmodels
        - cuml: Use cuML for GPU-accelerated logistic regression
    covariates
        Columns in adata.obs to include as covariates
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
    log2fc_threshold
        Minimum absolute log2 fold change to consider
    min_samples
        Minimum number of samples per condition level
    multitest_method
        Method for multiple testing correction
    n_jobs
        Number of parallel jobs
    batch_size
        Number of features to process per batch
    optimizer
        Optimizer to use for certain JAX methods.
        - BFGS: BFGS optimizer throgh jax.scipy.optimize
        - IRLS: Iteratively reweighted least squares (experimental)
    maxiter
        Maximum number of iterations for optimization
    verbose
        Whether to print progress messages

    Returns
    -------
    pandas.DataFrame
        Differential expression results
    """
    # Validate inputs
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")
    if covariates is not None:
        for col in covariates:
            if col not in adata.obs.columns:
                raise ValueError(f"Covariate '{col}' not found in adata.obs")

    # Get condition values
    condition_values = adata.obs[condition_key].values
    levels, comparisons = _validate_conditions(condition_values, reference, mode)

    # Check if grouping requested
    if group_key is not None:
        if group_key not in adata.obs.columns:
            raise ValueError(f"Group by key '{group_key}' not found in adata.obs")
        return grouped_de(
            adata=adata,
            condition_key=condition_key,
            reference=reference,
            group_key=group_key,
            method=method,
            backend=backend,
            covariates=covariates,
            mode=mode,
            layer=layer,
            data_type=data_type,
            log2fc_threshold=log2fc_threshold,
            min_samples=min_samples,
            multitest_method=multitest_method,
            n_jobs=n_jobs,
            batch_size=batch_size,
            optimizer=optimizer,
            maxiter=maxiter,
            verbose=verbose,
        )

    # Get expression matrix
    X = _get_layer(adata, layer)

    # Infer data type if auto
    if data_type == "auto":
        data_type = _infer_data_type(X)
        if verbose:
            print(f"Inferred data type: {data_type}")
    elif verbose:
        print(f"Using specified data type: {data_type}")

    # Validate method and data type combinations
    if method == "deseq2" and data_type != "counts":
        raise ValueError(f"DESeq2 requires count data. Current data type is {data_type}.")
    elif method.startswith("negbinom") and data_type != "counts":
        raise ValueError(f"Negative binomial models require count data. Current data type is {data_type}.")
    elif method.startswith("binomial") and data_type != "binary":
        raise ValueError(f"Binomial models require binary data. Current data type is {data_type}.")
    elif method.startswith("lr") and data_type not in ["lognorm", "binary"]:
        warnings.warn(
            "Logistic regression is designed for log-normalized or binary data. "
            f"Current data type is {data_type}, which may give unreliable results.",
            stacklevel=2,
        )
    elif (method.startswith("anova") or method.startswith("anova_residual")) and data_type != "lognorm":
        warnings.warn(
            "ANOVA is designed for log-normalized data. "
            f"Current data type is {data_type}, which may give unreliable results.",
            stacklevel=2,
        )

    if backend not in ["jax", "statsmodels", "cuml"]:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends are 'jax', 'statsmodels', 'cuml'.")

    if method == "deseq2":
        # Run DESeq2
        return _run_deseq2(
            adata=adata,
            condition_key=condition_key,
            comparisons=comparisons,
            covariates=covariates,
            layer=layer,
            n_cpus=n_jobs,
            verbose=verbose,
        )

    # Run tests for each comparison
    results = []
    for group1, group2 in comparisons:
        # Get cell masks
        mask1 = adata.obs[condition_key].values == group1
        mask2 = adata.obs[condition_key].values == group2

        if np.sum(mask1) < min_samples or np.sum(mask2) < min_samples:
            if verbose:
                warnings.warn(f"Skipping comparison {group1} vs {group2} with < {min_samples} samples", stacklevel=2)
            continue

        all_mask = mask1 | mask2

        # Get data for tests
        data = X[all_mask, :]
        model_data = _prepare_model_data(
            adata[all_mask, :],
            condition_key=condition_key,
            reference=group2,
            covariates=covariates,
        )
        condition_mask = model_data[condition_key].values == 1

        log2fc = _log2fc(
            X=data,
            condition_mask=condition_mask,
            data_type=data_type,
        )
        # Apply log2fc threshold
        mask = np.abs(log2fc) > log2fc_threshold
        data = data[:, mask]
        feature_names = adata.var_names[mask]

        if verbose:
            print(f"{np.sum(mask)} features passed log2fc threshold of {log2fc_threshold}")

        if backend == "jax":
            # Run batched DE test
            group_results = _run_batched_de(
                X=data,
                model_data=model_data,
                feature_names=feature_names,
                method=method,
                condition_key=condition_key,
                covariates=covariates,
                batch_size=batch_size,
                optimizer=optimizer,
                maxiter=maxiter,
                verbose=verbose,
            )

        else:
            # Run test per gene
            group_results = _run_de(
                X=data,
                model_data=model_data,
                feature_names=feature_names,
                method=method,
                backend=backend,
                condition_key=condition_key,
                covariates=covariates,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        auroc = _batched_auroc(
            X=data,
            groups=model_data[condition_key].values,
            batch_size=batch_size,
        )
        auroc_df = pd.DataFrame(
            {
                "feature": feature_names,
                "auroc": auroc,
            }
        )

        # Add comparison info and rename coef to log2fc
        group_results["feature"] = group_results["feature"].astype(str)
        group_results["test_condition"] = group1
        group_results["ref_condition"] = group2
        logfc_df = pd.DataFrame(
            {
                "log2fc": log2fc[mask],
                "feature": feature_names,
            }
        )
        group_results = group_results.merge(
            logfc_df,
            on="feature",
            how="left",
        ).merge(
            auroc_df,
            on="feature",
            how="left",
        )
        results.append(group_results)

    results = pd.concat(results, axis=0)

    # Check if any valid comparisons were found (length > 0 and not all pvals are NaN)
    if len(results) == 0 or results["pval"].isna().all():
        raise ValueError("No valid comparisons found for differential expression analysis")

    # Combine results and add multiple testing correction
    padj = sm.stats.multipletests(
        results["pval"][results["pval"].notna()],
        method=multitest_method,
    )[1]
    padj_df = pd.DataFrame(
        {
            "feature": results["feature"][results["pval"].notna()],
            "padj": padj,
        }
    )
    results = results.merge(
        padj_df,
        on="feature",
        how="left",
    )

    # Reorder columns
    return results[["feature", "test_condition", "ref_condition", "log2fc", "auroc", "coef", "pval", "padj"]]


def grouped_de(
    adata: AnnData,
    condition_key: str,
    group_key: str,
    reference: str | tuple[str, str] | None = None,
    method: Method = "deseq2",
    backend: str = "statsmodels",
    covariates: list[str] | None = None,
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    data_type: DataType = "auto",
    log2fc_threshold: float = 0.2,
    min_samples: int = 2,
    multitest_method: str = "fdr_bh",
    n_jobs: int = 1,
    batch_size: int = 32,
    optimizer: str = "BFGS",
    maxiter: int = 100,
    verbose: bool = False,
):
    """Run differential expression between condition levels grouped by cell type.

    Parameters
    ----------
    adata
        AnnData object
    condition_key
        Column in adata.obs containing condition values
    group_key
        Column in adata.obs for grouped DE testing
    reference
        Reference condition level or tuple (reference, comparison)
    method
        Testing method to use. One of:
        - deseq2: DESeq2 for count data
        - negbinom: Negative binomial GLM and wald test
        - lr: Logistic regression and likelihood ratio test
        - anova: ANOVA based on a linear model
        - anova_residual: Linear model with residual F-test
        - binomial: Binomial GLM
    backend
        Backend to use for linear model-based DE methods.
        - jax: Use custom linear models in JAX for batched and GPU-accelerated methods
        - statsmodels: Use linear models from statsmodels
        - cuml: Use cuML for GPU-accelerated logistic regression
    covariates
        Columns in adata.obs to include as covariates
    mode
        How to perform comparisons
    layer
        Layer in adata to use
    data_type
        Type of data:
        - counts: Raw count data
        - lognorm: Log-normalized data (assumed to be log1p of normalized counts)
        - binary: Binary data
        - auto: Automatically infer data type
    log2fc_threshold
        Minimum absolute log2 fold change to consider
    min_samples
        Minimum number of samples per condition level
    multitest_method
        Method for multiple testing correction
    n_jobs
        Number of parallel jobs
    batch_size
        Number of features to process per batch
    optimizer
        Optimizer for model fitting (only used for certain JAX methods)
    maxiter
        Maximum number of iterations for optimization
    verbose
        Whether to print progress messages

    Returns
    -------
    pandas.DataFrame
        Differential expression results grouped by cell type
    """
    results = []
    for group in adata.obs[group_key].unique():
        mask = adata.obs[group_key].values == group
        if sum(mask) < (min_samples * 2):
            if verbose:
                warnings.warn(f"Skipping group {group} with < {min_samples * 2} samples", stacklevel=2)
            continue

        # Run DE for group
        group_results = de(
            adata=adata[mask, :],
            condition_key=condition_key,
            reference=reference,
            group_key=None,
            method=method,
            covariates=covariates,
            mode=mode,
            layer=layer,
            data_type=data_type,
            log2fc_threshold=log2fc_threshold,
            min_samples=min_samples,
            multitest_method=multitest_method,
            n_jobs=n_jobs,
            batch_size=batch_size,
            optimizer=optimizer,
            maxiter=maxiter,
            verbose=verbose,
        )
        group_results["group"] = group
        results.append(group_results)

    # Combine results and add multiple testing correction
    if len(results) == 0:
        raise ValueError("No valid groups found for differential expression analysis")

    results = pd.concat(results, axis=0)
    results["padj"] = sm.stats.multipletests(
        results["pval"],
        method=multitest_method,
    )[1]

    return results
