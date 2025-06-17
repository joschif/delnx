"""Batched differential expression test functions in JAX."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import patsy
import scipy.stats as stats
import tqdm
from scipy import sparse

from delnx._utils import _to_dense
from delnx.models import LinearRegression, LogisticRegression, NegativeBinomialRegression


@partial(jax.jit, static_argnums=(3, 4))
def _fit_lr(y, covars, x=None, optimizer="BFGS", maxiter=100):
    """Fit single logistic regression model with JAX."""
    model = LogisticRegression(skip_wald=True, optimizer=optimizer, maxiter=maxiter)

    # Covars should include intercept
    if x is not None:
        X = jnp.column_stack([covars, x])
    else:
        X = covars

    results = model.fit(X, y)

    ll = results["llf"]
    coefs = results["coef"]

    return ll, coefs


_fit_lr_batch = jax.vmap(_fit_lr, in_axes=(None, None, 1, None, None), out_axes=(0, 0))


def _run_lr_test(
    X: jnp.ndarray,
    cond: jnp.ndarray,
    covars: jnp.ndarray | None = None,
    optimizer: str = "BFGS",
    maxiter: int = 100,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run logistic regression test for a batch of features.

    Args:
        X: Expression data for current batch, shape (n_cells, batch_size)
        cond: Binary condition labels, shape (n_cells,)
        covars: Covariate data with intercept, shape (n_cells, n_covariates)
        optimizer: Optimization algorithm to use
        maxiter: Maximum number of iterations for optimization

    Returns
    -------
        Tuple of coefficients and p-values for the batch
    """
    # Fit null model (with intercept only)
    ll_null, _ = _fit_lr(cond, covars, optimizer=optimizer, maxiter=maxiter)

    # Vectorized fit of full models
    ll_full, coefs_full = _fit_lr_batch(cond, covars, X, optimizer, maxiter)

    # Vectorized computation of test statistics
    lr_stats = 2 * (ll_full - ll_null)
    pvals = stats.chi2.sf(lr_stats, 1)

    # Extract coefficients (second parameter is the feature coefficient)
    coefs = coefs_full[:, -1]

    return coefs, pvals


@partial(jax.jit, static_argnums=(5, 6, 7))
def _fit_nb(x, y, covars, disp, size_factors=None, optimizer="BFGS", maxiter=100, dispersion_method="mle"):
    """Fit single negative binomial regression model with JAX."""
    model = NegativeBinomialRegression(
        dispersion=disp, optimizer=optimizer, maxiter=maxiter, dispersion_method=dispersion_method
    )

    # Covars should already include intercept
    X = jnp.column_stack([covars, x])
    offset = jnp.log(size_factors) if size_factors is not None else None
    results = model.fit(X, y, offset=offset)

    coefs = results["coef"]
    pvals = results["pval"]

    return coefs, pvals


def _run_nb_test(
    X: jnp.ndarray,
    cond: jnp.ndarray,
    covars: jnp.ndarray,
    disp: jnp.ndarray | None = None,
    size_factors: jnp.ndarray | None = None,
    optimizer: str = "BFGS",
    maxiter: int = 100,
    dispersion_method: str = "mle",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run negative binomial regression test for a batch of features.

    Args:
        X: Expression data for current batch, shape (n_cells, batch_size)
        cond: Binary condition labels, shape (n_cells,)
        covars: Covariate data with intercept, shape (n_cells, n_covariates)
        disp: Dispersion parameters for each feature, shape (batch_size,)
        size_factors: Size factors for normalization, shape (n_cells,), optional
        optimizer: Optimization algorithm to use
        maxiter: Maximum number of iterations for optimization
        dispersion_method: Method to estimate gene-wise dispersions if not provided
            - "mle": Maximum likelihood estimation
            - "moments": Method of moments

    Returns
    -------
        Tuple of coefficients and p-values for the batch
    """

    def fit_nb(x, disp):
        return _fit_nb(
            cond,
            x,
            covars,
            disp,
            size_factors=size_factors,
            optimizer=optimizer,
            maxiter=maxiter,
            dispersion_method=dispersion_method,
        )

    if disp is None:
        fit_nb_batch = jax.vmap(fit_nb, in_axes=(1, None), out_axes=(0, 0))
    else:
        fit_nb_batch = jax.vmap(fit_nb, in_axes=(1, 0), out_axes=(0, 0))
        disp = disp.reshape(-1, 1) if disp.ndim == 1 else disp

    coefs, pvals = fit_nb_batch(X, disp)
    return coefs[:, -1], pvals[:, -1]


@partial(jax.jit, static_argnums=(3, 4))
def _fit_anova(x, y, covars, method="anova", maxiter=100):
    """Fit linear model based ANOVA."""
    model = LinearRegression(skip_wald=True, maxiter=maxiter)
    n = y.shape[0]

    # Fit null model (without feature)
    X_null = covars
    results_null = model.fit(X_null, y)
    # Null model predictions and residuals
    pred_null = X_null @ results_null["coef"]
    ss_null = jnp.sum((y - pred_null) ** 2)
    df_null = n - X_null.shape[1]
    ms_null = ss_null / df_null

    # Fit full model (with feature)
    X_full = jnp.column_stack([covars, x])
    results_full = model.fit(X_full, y)
    coef = results_full["coef"][-1]  # Feature coefficient
    # Full model predictions and residuals
    pred_full = X_full @ results_full["coef"]
    ss_full = jnp.sum((y - pred_full) ** 2)
    df_full = n - X_full.shape[1]
    ms_full = ss_full / df_full

    # Calculate F-statistic and p-value
    if method == "anova":
        # Calculate sum of squares due to the feature
        ss_feature = ss_null - ss_full
        df_feature = 1  # One feature added to null model
        # F-statistic
        ms_feature = ss_feature / df_feature
        f_stat = ms_feature / ms_full
        return coef, (f_stat, df_feature, df_full)

    else:  # residual test
        # Compare residual variance between models
        f_stat = ms_null / ms_full
        return coef, (f_stat, df_null, df_full)


_fit_anova_batch = jax.vmap(_fit_anova, in_axes=(None, 1, None, None, None), out_axes=(0, 0))


def _run_anova_test(
    X: jnp.ndarray,
    cond: jnp.ndarray,
    covars: jnp.ndarray,
    method: str = "anova",
    maxiter: int = 100,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run ANOVA test for a batch of features.
    Args:
        X: Expression data for current batch, shape (n_cells, batch_size)
        cond: Binary condition labels, shape (n_cells,)
        covars: Optional covariate data, shape (n_cells, n_covariates)
        method: Type of test, either "anova" or "residual"
        maxiter: Maximum number of iterations for optimization

    Returns
    -------
        Tuple of coefficients and p-values for the batch
    """  # noqa: D205
    coefs, (f_stat, dfn, dfd) = _fit_anova_batch(cond, X, covars, method, maxiter)

    if method == "anova":
        pvals = stats.f.sf(f_stat, dfn, dfd)

    else:  # residual test
        p_resid_cdf = stats.f.cdf(f_stat, dfn, dfd)
        pvals = 1 - np.abs(0.5 - p_resid_cdf) * 2

    return coefs, pvals


def _run_batched_de(
    X: np.ndarray | sparse.spmatrix,
    model_data: pd.DataFrame,
    feature_names: pd.Index,
    method: str,
    condition_key: str,
    dispersions: np.ndarray | None = None,
    size_factors: np.ndarray | None = None,
    covariate_keys: list[str] | None = None,
    dispersion_method: str = "mle",
    batch_size: int = 32,
    optimizer: str = "BFGS",
    maxiter: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run differential expression analysis in batches.

    Parameters
    ----------
        X: Expression data matrix, shape (n_cells, n_features)
        model_data: DataFrame containing condition and covariate data
        feature_names: Names of features/genes
        method: Testing method to use:
            - "lr": Logistic regression with LR test
            - "negbinom": Negative binomial with Wald test
            - "anova": Linear model with ANOVA F-test
            - "anova_residual": Linear model with residual F-test
        condition_key: Name of condition column in model_data
        optimizer: Optimization algorithm to use
        maxiter: Maximum number of iterations for optimization
        dispersions: Dispersion estimates for negative binomial regression, shape (n_features,)
        dispersion_method: Method to estimate gene-wise dispersions if not provided
            - "mle": Maximum likelihood estimation
            - "moments": Method of moments
        size_factors: Size factors for normalization, shape (n_cells,)
        covariate_keys: Names of covariate columns in model_data
        batch_size: Number of features to process per batch
        verbose: Whether to show progress bar

    Returns
    -------
        DataFrame with test results for each feature
    """
    # Prepare data for logistic regression
    if method == "lr":
        conditions = jnp.asarray(model_data[condition_key].values, dtype=jnp.float32)
        covars = patsy.dmatrix(" + ".join(covariate_keys), model_data) if covariate_keys else np.ones((X.shape[0], 1))
        covars = jnp.asarray(covars, dtype=jnp.float32)

        def test_fn(x):
            return _run_lr_test(x, conditions, covars, optimizer=optimizer, maxiter=maxiter)

    # Prepare data for negative binomial regression
    elif method == "negbinom":
        conditions = jnp.asarray(model_data[condition_key].values, dtype=jnp.float32)
        covars = patsy.dmatrix(" + ".join(covariate_keys), model_data) if covariate_keys else np.ones((X.shape[0], 1))
        covars = jnp.asarray(covars, dtype=jnp.float32)

        def test_fn(x, disp=None):
            return _run_nb_test(
                x,
                conditions,
                covars,
                disp,
                size_factors=size_factors,
                optimizer=optimizer,
                maxiter=maxiter,
                dispersion_method=dispersion_method,
            )

    # Prepare data for ANOVA tests
    elif method in ["anova", "anova_residual"]:
        conditions = jnp.asarray(model_data[condition_key].values, dtype=jnp.float32)
        covars = patsy.dmatrix(" + ".join(covariate_keys), model_data) if covariate_keys else np.ones((X.shape[0], 1))
        covars = jnp.asarray(covars, dtype=jnp.float32)
        anova_method = "anova" if method == "anova" else "residual"

        def test_fn(x):
            return _run_anova_test(x, conditions, covars, anova_method, maxiter=maxiter)

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Process run DE tests in batches
    n_features = X.shape[1]
    results = {
        "feature": [],
        "coef": [],
        "pval": [],
    }
    for i in tqdm.tqdm(range(0, n_features, batch_size), disable=not verbose):
        batch = slice(i, min(i + batch_size, n_features))
        X_batch = jnp.asarray(_to_dense(X[:, batch]), dtype=jnp.float32)

        if method == "negbinom" and dispersions is not None:
            disp_batch = jnp.asarray(dispersions[batch], dtype=jnp.float32)
            coefs, pvals = test_fn(X_batch, disp_batch)
        else:
            coefs, pvals = test_fn(X_batch)

        results["feature"].extend(feature_names[batch].tolist())
        results["coef"].extend(coefs.tolist())
        results["pval"].extend(pvals.tolist())

    return pd.DataFrame(results)
