"""Batched differential expression testing with JAX."""

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
    """Fit a single logistic regression model using JAX.

    This function fits a logistic regression model for a single feature,
    with support for covariates. It is designed to be used within a batched
    context via JAX's vmap functionality.

    Parameters
    ----------
    y : jnp.ndarray
        Binary outcome variable of shape (n_samples,).
    covars : jnp.ndarray
        Covariate matrix including intercept of shape (n_samples, n_covariates).
    x : jnp.ndarray, optional
        Feature values of shape (n_samples,). If None, only the null model with
        covariates is fitted.
    optimizer : str, default='BFGS'
        Optimization method to use for fitting the model.
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.

    Returns
    -------
    tuple
        Log-likelihood value and coefficient estimates.
    """
    model = LogisticRegression(skip_stats=True, optimizer=optimizer, maxiter=maxiter)

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

    This function performs logistic regression-based differential expression
    testing for multiple features in parallel. It fits both null and alternative
    models, and computes likelihood ratio test statistics and p-values.

    Parameters
    ----------
    X : jnp.ndarray
        Expression data matrix of shape (n_samples, n_features), where each
        column represents a feature (gene) to be tested.
    cond : jnp.ndarray
        Binary condition labels of shape (n_samples,), typically encoding
        treatment or control groups.
    covars : jnp.ndarray | None, default=None
        Covariate data including intercept of shape (n_samples, n_covariates).
        Should include a column of ones for the intercept.
    optimizer : str, default='BFGS'
        Optimization method to use for model fitting.
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing:
        - Feature coefficients of shape (n_features,)
        - P-values from likelihood ratio test of shape (n_features,)
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
    """Fit a single negative binomial regression model using JAX.

    This function fits a negative binomial regression model for a single feature,
    with support for covariates and size factors for normalization. It is designed
    to be used within batched operations.

    Parameters
    ----------
    x : jnp.ndarray
        Feature values of shape (n_samples,).
    y : jnp.ndarray
        Count data outcome variable of shape (n_samples,).
    covars : jnp.ndarray
        Covariate matrix including intercept of shape (n_samples, n_covariates).
    disp : float | None
        Fixed dispersion parameter. If None, it will be estimated from the data.
    size_factors : jnp.ndarray | None, default=None
        Size factors for normalization of shape (n_samples,). Will be log-transformed
        and used as offset in the model.
    optimizer : str, default='BFGS'
        Optimization method to use for fitting the model.
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.
    dispersion_method : str, default='mle'
        Method for estimating dispersion if not provided. Options are 'mle' or 'moments'.

    Returns
    -------
    tuple
        Coefficient estimates and p-values from the model.
    """
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
    """Run negative binomial regression tests for a batch of features.

    This function performs negative binomial regression-based differential
    expression testing for multiple features in parallel. It leverages the
    NegativeBinomialRegression model and supports size factor normalization
    through offset terms.

    Parameters
    ----------
    X : jnp.ndarray
        Expression data matrix of shape (n_samples, n_features), where each
        column represents a feature (gene) to be tested.
    cond : jnp.ndarray
        Condition indicator of shape (n_samples,). Should be a design vector
        where each element indicates group membership or a continuous covariate.
    covars : jnp.ndarray
        Covariate matrix including intercept of shape (n_samples, n_covariates).
        Must include a column of ones for the intercept.
    disp : jnp.ndarray | None, default=None
        Dispersion parameters for each feature of shape (n_features,).
        If None, dispersion will be estimated for each feature.
    size_factors : jnp.ndarray | None, default=None
        Size factors for normalization of shape (n_samples,).
        These will be log-transformed and used as offset in the model.
    optimizer : str, default='BFGS'
        Optimization method to use for model fitting.
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.
    dispersion_method : str, default='mle'
        Method to estimate gene-wise dispersions if not provided:
        - 'mle': Maximum likelihood estimation (more accurate but slower)
        - 'moments': Method of moments (faster but less accurate)

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing:
        - Feature coefficients of shape (n_features,)
        - P-values from Wald tests of shape (n_features,)
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
    """Fit linear model-based ANOVA for a single feature.

    This function fits linear models and performs ANOVA-type analysis
    to test for differential expression of a continuous feature.
    It compares a null model (covariates only) to a full model
    (covariates + feature) using an F-test.

    Parameters
    ----------
    x : jnp.ndarray
        Feature values of shape (n_samples,).
    y : jnp.ndarray
        Continuous response variable of shape (n_samples,).
    covars : jnp.ndarray
        Covariate matrix including intercept of shape (n_samples, n_covariates).
    method : str, default='anova'
        Method for the test. Currently only 'anova' is supported.
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.

    Returns
    -------
    tuple
        Coefficient estimate and p-value from F-test.
    """
    model = LinearRegression(skip_stats=True, maxiter=maxiter)
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
    """Run ANOVA-based tests for a batch of features.

    This function performs linear model-based ANOVA tests for
    multiple features in parallel. It can perform standard ANOVA
    (testing for the effect of adding a feature) or residual F-tests
    (testing for differences in residual variances).

    Parameters
    ----------
    X : jnp.ndarray
        Expression data matrix of shape (n_samples, n_features), where each
        column represents a feature (gene) to be tested.
    cond : jnp.ndarray
        Continuous response variable of shape (n_samples,).
    covars : jnp.ndarray
        Covariate matrix including intercept of shape (n_samples, n_covariates).
    method : str, default='anova'
        Type of test to perform:
        - 'anova': Standard ANOVA F-test for feature effect
        - 'residual': Residual F-test comparing error variances
    maxiter : int, default=100
        Maximum number of iterations for the optimizer.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Tuple containing:
        - Feature coefficients of shape (n_features,)
        - P-values from F-tests of shape (n_features,)
    """
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

    This function is the main entry point for performing differential expression
    analysis using JAX-based implementations. It processes large expression matrices
    in batches to optimize memory usage and leverages JAX for acceleration.
    The function supports different statistical methods and handles various
    modeling approaches including offset terms for size factor normalization.

    Parameters
    ----------
    X : np.ndarray | sparse.spmatrix
        Expression data matrix of shape (n_samples, n_features).
    model_data : pd.DataFrame
        DataFrame containing condition labels and covariates.
    feature_names : pd.Index
        Names of features/genes corresponding to columns in X.
    method : str
        Statistical method for testing:
        - 'lr': Logistic regression with likelihood ratio test
        - 'negbinom': Negative binomial regression with Wald test
        - 'anova': Linear model with ANOVA F-test
        - 'anova_residual': Linear model with residual F-test
    condition_key : str
        Name of the column in model_data containing condition labels.
    dispersions : np.ndarray | None, default=None
        Pre-computed dispersion estimates for negative binomial regression
        of shape (n_features,).
    size_factors : np.ndarray | None, default=None
        Size factors for normalization of shape (n_samples,). Will be
        log-transformed and used as offset in the model.
    covariate_keys : list[str] | None, default=None
        Names of covariate columns in model_data to include in the design matrix.
    dispersion_method : str, default='mle'
        Method for estimating gene-wise dispersions for negative binomial models if not provided:
        - 'mle': Maximum likelihood estimation based on the intercept
        - 'moments': Method of moments
    batch_size : int, default=32
        Number of features to process in each batch for memory efficiency.
    optimizer : str, default='BFGS'
        Optimization algorithm for fitting models.
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    verbose : bool, default=True
        Whether to display progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame with test results for each feature, including:
        - Feature names
        - Coefficients/effect sizes
        - P-values
        - Other test-specific statistics
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
