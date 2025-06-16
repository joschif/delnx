"""Differential expression testing module."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData

from delnx._typing import Backends, ComparisonMode, DataType, Method
from delnx._utils import _get_layer

from ._de_tests import _run_de, _run_deseq2
from ._effects import _batched_auroc, _log2fc
from ._jax_tests import _run_batched_de
from ._utils import (
    _infer_data_type,
    _prepare_model_data,
    _validate_conditions,
)

SUPPORTED_BACKENDS = ["jax", "statsmodels", "cuml"]


def _grouped_de(
    adata: AnnData,
    condition_key: str,
    group_key: str,
    reference: str | tuple[str, str] | None = None,
    size_factor_key: str | None = None,
    dispersion_key: str | None = None,
    covariate_keys: list[str] | None = None,
    method: Method = "lr",
    backend: Backends = "jax",
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    data_type: DataType = "auto",
    log2fc_threshold: float = 0.0,
    min_samples: int = 2,
    multitest_method: str = "fdr_bh",
    n_jobs: int = 1,
    batch_size: int = 2048,
    optimizer: str = "BFGS",
    maxiter: int = 100,
    verbose: bool = True,
):
    """
    Perform differential expression analysis for each group separately.
    This function runs DE tests for each unique group in `adata.obs[group_key]`
    and combines the results.
    """  # noqa: D205
    results = []
    for group in adata.obs[group_key].unique():
        mask = adata.obs[group_key].values == group
        if sum(mask) < (min_samples * 2):
            if verbose:
                warnings.warn(
                    f"Skipping group {group} with < {min_samples * 2} samples",
                    stacklevel=2,
                )
            continue

        # Run DE for group
        group_results = de(
            adata=adata[mask, :],
            condition_key=condition_key,
            reference=reference,
            group_key=None,
            method=method,
            backend=backend,
            size_factor_key=size_factor_key,
            dispersion_key=dispersion_key,
            covariate_keys=covariate_keys,
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

    results = pd.concat(results, axis=0).reset_index(drop=True)

    # Check if any results are valid
    if len(results) == 0 or results["pval"].isna().all():
        raise ValueError(
            "Differential expression analysis failed for all groups. Please check the input data or set `verbose=True` for more details."
        )

    # Perform multiple testing correction
    padj = sm.stats.multipletests(
        results["pval"][results["pval"].notna()].values,
        method=multitest_method,
    )[1]
    results["padj"] = np.nan  # Initialize with NaN
    results.loc[results["pval"].notna(), "padj"] = padj

    results = results.sort_values(
        by=["test_condition", "ref_condition", "padj"],
    ).reset_index(drop=True)

    # Reorder columns
    return results[
        [
            "feature",
            "test_condition",
            "ref_condition",
            "log2fc",
            "auroc",
            "coef",
            "pval",
            "padj",
        ]
    ]


def de(
    adata: AnnData,
    condition_key: str,
    group_key: str | None = None,
    reference: str | tuple[str, str] | None = None,
    size_factor_key: str | None = None,
    dispersion_key: str | None = None,
    covariate_keys: list[str] | None = None,
    method: Method = "lr",
    backend: Backends = "jax",
    mode: ComparisonMode = "all_vs_all",
    layer: str | None = None,
    data_type: DataType = "auto",
    log2fc_threshold: float = 0.0,
    min_samples: int = 2,
    multitest_method: str = "fdr_bh",
    n_jobs: int = 1,
    batch_size: int = 2048,
    optimizer: str = "BFGS",
    maxiter: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform differential expression analysis between condition levels.

    This function runs differential expression testing using various statistical methods
    and backends. It supports both single and grouped comparisons with multiple
    testing correction.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing expression data and metadata.
    condition_key : str
        Column name in `adata.obs` containing condition labels for comparison.
    group_key : str, optional
        Column name in `adata.obs` for grouped differential expression testing
        (e.g., cell type). If provided, DE testing is performed separately for
        each group. Default is None.
    reference : str or tuple of str, optional
        Reference condition for comparison. Can be:
        - Single string: reference condition for all comparisons
        - Tuple (reference, comparison): specific pair to compare
        - None: automatically determined based on mode
        Default is None.
    size_factor_key : str, optional
        Column name in `adata.obs` containing size factors for normalization.
        Used as offset in negative binomial models. Default is None.
    dispersion_key : str, optional
        Column name in `adata.var` containing precomputed dispersions.
        Only used for negative binomial methods. Default is None.
    covariate_keys : list of str, optional
        List of column names in `adata.obs` to include as covariates in the model.
        Default is None.
    method : {"lr", "deseq2", "negbinom", "anova", "anova_residual", "binomial"}
        Statistical method for differential expression testing:
        - "lr": Logistic regression with likelihood ratio test
        - "deseq2": DESeq2 method for count data (requires pydeseq2)
        - "negbinom": Negative binomial GLM with Wald test
        - "anova": ANOVA based on linear model
        - "anova_residual": Linear model with residual F-test
        - "binomial": Binomial GLM
        Default is "lr".
    backend : {"jax", "statsmodels", "cuml"}
        Computational backend for linear model-based methods:
        - "jax": Custom JAX implementation (batched, GPU-accelerated)
        - "statsmodels": Standard statsmodels implementation
        - "cuml": cuML for GPU-accelerated logistic regression
        Default is "jax".
    mode : {"all_vs_all", "all_vs_ref", "1_vs_1"}
        Comparison strategy:
        - "all_vs_all": Compare all pairs of condition levels
        - "all_vs_ref": Compare all levels against reference
        - "1_vs_1": Compare only reference vs comparison (requires tuple reference)
        Default is "all_vs_all".
    layer : str, optional
        Layer name in `adata.layers` to use for expression data.
        If None, uses `adata.X`. Default is None.
    data_type : {"auto", "counts", "lognorm", "binary"}
        Type of expression data:
        - "auto": Automatically infer from data
        - "counts": Raw count data
        - "lognorm": Log-normalized data (log1p of normalized counts)
        - "binary": Binary expression data
        Default is "auto".
    log2fc_threshold : float
        Minimum absolute log2 fold change threshold for feature inclusion.
        Features below this threshold are excluded from testing. Default is 0.0.
    min_samples : int
        Minimum number of samples required per condition level.
        Comparisons with fewer samples are skipped. Default is 2.
    multitest_method : str
        Method for multiple testing correction. Accepts any method supported
        by `statsmodels.stats.multipletests`. Common options include:
        - "fdr_bh": Benjamini-Hochberg FDR correction
        - "bonferroni": Bonferroni correction
        Default is "fdr_bh".
    n_jobs : int
        Number of parallel jobs for non-JAX backends. Default is 1.
    batch_size : int
        Number of features to process per batch. Reduce for memory-constrained
        environments or very large datasets (>1M samples). Default is 2048.
    optimizer : {"BFGS", "IRLS"}
        Optimization algorithm for JAX backend:
        - "BFGS": BFGS optimizer via jax.scipy.optimize
        - "IRLS": Iteratively reweighted least squares (experimental)
        Default is "BFGS".
    maxiter : int
        Maximum number of optimization iterations. Default is 100.
    verbose : bool
        Whether to print progress messages and warnings. Default is True.

    Returns
    -------
    pd.DataFrame
        Differential expression results with columns:
        - "feature": Feature/gene names
        - "test_condition": Test condition label
        - "ref_condition": Reference condition label
        - "log2fc": Log2 fold change (test vs reference)
        - "auroc": Area under ROC curve
        - "coef": Model coefficient
        - "pval": Raw p-value
        - "padj": Adjusted p-value (multiple testing corrected)
        - "group": Group label (only for grouped DE)

    Raises
    ------
    ValueError
        If required keys are missing from adata.obs/var, if method and data_type
        are incompatible, or if no valid comparisons are found.

    Examples
    --------
    Basic differential expression between two conditions:

    >>> results = de(adata, condition_key="treatment", reference="control", mode="all_vs_ref")

    Pairwise DE testing between all conditions:
    >>> results = de(adata, condition_key="treatment", mode="all_vs_all")

    Grouped DE by cell type with covariates:

    >>> results = de(
    ...     adata, condition_key="disease_state", group_key="cell_type", covariate_keys=["age", "sex"], method="deseq2"
    ... )

    Testing on count data with a negative binomial model:

    >>> results = de(
    ...     adata,
    ...     condition_key="condition",
    ...     method="negbinom",
    ...     size_factor_key="size_factors",
    ...     dispersion_key="dispersions",
    ...     layer="counts",
    ... )

    Using other methods and backends:
    >>> results = de(adata, method="binomial", backend="statsmodels", condition_key="treatment")
    >>> results = de(adata, method="anova", condition_key="treatment", data_type="lognorm")
    >>> results = de(adata, method="lr", backend="cuml", condition_key="treatment")
    >>> results = de(adata, method="deseq2", condition_key="treatment", layer="counts")

    Notes
    -----
    - Method and data type compatibility:
      - "deseq2" and "negbinom" require count data
      - "binomial" requires binary data
      - "lr" works best with log-normalized or binary data
      - "anova" methods work best with log-normalized data
    - Backend options:
        - "jax" provides batched, GPU-accelerated testing with the following methods: "lr", "negbinom", "anova", "anova_residual"
        - "statsmodels" uses statsmodels implementations of regression models with the following methods: "lr", "negbinom", "anova", "anova_residual", "binomial"
        - "cuml" provides GPU-accelerated logistic regression with the "lr" method
    - The "deseq2" method ignores the `backend` parameter and always uses the PyDESeq2 implementation.
    - Size factors and dispersion parameters shoudl be pre-computed for the "negbinom" method.
    - Multiple testing correction is applied across all comparisons
    """
    # Validate inputs
    if condition_key not in adata.obs.columns:
        raise ValueError(f"Condition key '{condition_key}' not found in adata.obs")
    if covariate_keys is not None:
        for col in covariate_keys:
            if col not in adata.obs.columns:
                raise ValueError(f"Covariate '{col}' not found in adata.obs")
    if size_factor_key is not None and size_factor_key not in adata.obs.columns:
        raise ValueError(f"Size factors key '{size_factor_key}' not found in adata.obs")
    if dispersion_key is not None and dispersion_key not in adata.var.columns:
        raise ValueError(f"Dispersions key '{dispersion_key}' not found in adata.var")

    # Get condition values
    condition_values = adata.obs[condition_key].values
    # Get size factors and dispersions if provided
    size_factors = adata.obs[size_factor_key].values if size_factor_key else None
    dispersions = adata.var[dispersion_key].values if dispersion_key else None

    # Validate conditions and get comparison levels
    levels, comparisons = _validate_conditions(condition_values, reference, mode)

    # Check if grouping requested
    if group_key is not None:
        if group_key not in adata.obs.columns:
            raise ValueError(f"Group by key '{group_key}' not found in adata.obs")
        return _grouped_de(
            adata=adata,
            condition_key=condition_key,
            reference=reference,
            group_key=group_key,
            size_factor_key=size_factor_key,
            dispersion_key=dispersion_key,
            covariate_keys=covariate_keys,
            method=method,
            backend=backend,
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

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends are 'jax', 'statsmodels', 'cuml'.")

    if method == "deseq2":
        # Run PyDESeq2
        return _run_deseq2(
            adata=adata,
            condition_key=condition_key,
            comparisons=comparisons,
            covariate_keys=covariate_keys,
            multitest_method=multitest_method,
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
                warnings.warn(
                    f"Skipping comparison {group1} vs {group2} with < {min_samples} samples",
                    stacklevel=2,
                )
            continue

        all_mask = mask1 | mask2

        # Get data for tests
        X_comp = X[all_mask, :]
        sf_comp = size_factors[all_mask] if size_factors is not None else None

        if data_type == "counts" and size_factors is not None:
            X_norm = X_comp / sf_comp[:, np.newaxis]
        else:
            X_norm = X_comp

        model_data = _prepare_model_data(
            adata[all_mask, :],
            condition_key=condition_key,
            reference=group2,
            covariate_keys=covariate_keys,
        )
        condition_mask = model_data[condition_key].values == 1

        # Calculate log2 fold change
        log2fc = _log2fc(X=X_norm, condition_mask=condition_mask, data_type=data_type)

        # Apply log2fc threshold
        feature_mask = np.abs(log2fc) > log2fc_threshold
        X_comp = X_comp[:, feature_mask]
        X_norm = X_norm[:, feature_mask]
        feature_names = adata.var_names[feature_mask].values

        if verbose:
            print(f"{np.sum(feature_mask)} features passed log2fc threshold of {log2fc_threshold}")

        if backend == "jax":
            # Run batched DE test
            group_results = _run_batched_de(
                X=X_comp,
                model_data=model_data,
                feature_names=feature_names,
                method=method,
                condition_key=condition_key,
                dispersions=dispersions,
                size_factors=sf_comp,
                covariate_keys=covariate_keys,
                batch_size=batch_size,
                optimizer=optimizer,
                maxiter=maxiter,
                verbose=verbose,
            )

        else:
            # Run test per gene
            group_results = _run_de(
                X=X_comp,
                model_data=model_data,
                feature_names=feature_names,
                method=method,
                backend=backend,
                condition_key=condition_key,
                size_factors=size_factors,
                covariate_keys=covariate_keys,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        auroc = _batched_auroc(X=X_norm, groups=model_data[condition_key].values, batch_size=batch_size)
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
                "log2fc": log2fc[feature_mask],
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

    results = pd.concat(results, axis=0).reset_index(drop=True)

    # Check if any valid comparisons were found (length > 0 and not all pvals are NaN)
    if len(results) == 0 or results["pval"].isna().all():
        raise ValueError(
            "Differential expression analysis failed for all comparisons. Please check the input data or set `verbose=True` for more details."
        )

    # Perform multiple testing correction
    padj = sm.stats.multipletests(
        results["pval"][results["pval"].notna()].values,
        method=multitest_method,
    )[1]
    results["padj"] = np.nan  # Initialize with NaN
    results.loc[results["pval"].notna(), "padj"] = padj

    results = results.sort_values(
        by=["test_condition", "ref_condition", "padj"],
    ).reset_index(drop=True)

    # Reorder columns
    return results[
        [
            "feature",
            "test_condition",
            "ref_condition",
            "log2fc",
            "auroc",
            "coef",
            "pval",
            "padj",
        ]
    ]
