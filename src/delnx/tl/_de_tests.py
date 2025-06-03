"""Standard differential expression test functions."""

import warnings
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
from anndata import AnnData
from joblib import Parallel, delayed
from orchard._utils import _get_layer, _to_dense, suppress_output
from pydeseq2.dds import DefaultInference, DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy import sparse, stats
from sklearn.metrics import log_loss
from statsmodels.stats.anova import anova_lm


def _run_lr_test(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariates: list[str] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Run logistic regression with likelihood ratio test."""
    model_data = model_data.copy()
    model_data["X"] = X

    covar_str = ""
    if covariates:
        covar_str = " + " + " + ".join(covariates)

    # Fit models
    full_formula = f"{condition_key} ~ X{covar_str}"
    reduced_formula = f"{condition_key} ~ 1{covar_str}"

    with suppress_output(verbose):
        full_model = sm.Logit.from_formula(full_formula, data=model_data).fit(disp=False)
        reduced_model = sm.Logit.from_formula(reduced_formula, data=model_data).fit(disp=False)

    # LR test
    lr_stat = 2 * (full_model.llf - reduced_model.llf)
    df_diff = full_model.df_model - reduced_model.df_model
    pval = stats.chi2.sf(lr_stat, df_diff)

    return full_model.params["X"], pval


def _run_negbinom(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariates: list[str] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Run negative binomial GLM."""
    model_data = model_data.copy()
    model_data["X"] = X

    covar_str = ""
    if covariates:
        covar_str = " + " + " + ".join(covariates)

    formula = f"X ~ {condition_key}{covar_str}"

    # Fit model
    with suppress_output(verbose):
        model = sm.NegativeBinomial.from_formula(formula, data=model_data).fit(disp=False)

    return model.params[condition_key], model.pvalues[condition_key]


def _run_anova(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariates: list[str] | None = None,
    method: str = "anova",
    verbose: bool = False,
) -> tuple[float, float]:
    """Run ANOVA test."""
    model_data = model_data.copy()
    model_data["X"] = X

    covar_str = ""
    if covariates:
        covar_str = " + " + " + ".join(covariates)

    null_formula = f"X ~ 1{covar_str}"
    formula = f"X ~ {condition_key}{covar_str}"

    with suppress_output(verbose):
        null_model = sm.OLS.from_formula(null_formula, data=model_data).fit()
        model = sm.OLS.from_formula(formula, data=model_data).fit()
        a0 = anova_lm(null_model)
        a1 = anova_lm(model)

    p_anova = a1.loc[condition_key, "PR(>F)"]
    p_resid_cdf = stats.f.cdf(
        a0.loc["Residual", "mean_sq"] / a1.loc["Residual", "mean_sq"],
        a0.loc["Residual", "df"],
        a1.loc["Residual", "df"],
    )
    p_resid = 1 - np.abs(0.5 - p_resid_cdf) * 2

    return model.params[condition_key], p_anova if method == "anova" else p_resid


def _run_binomial(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariates: list[str] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Run binomial GLM."""
    model_data = model_data.copy()
    model_data["X"] = X

    covar_str = ""
    if covariates:
        covar_str = " + " + " + ".join(covariates)

    # Use X as response and condition as predictor
    formula = f"X ~ {condition_key}{covar_str}"

    with suppress_output(verbose):
        # Fit model with binomial family and logit link
        glm = sm.GLM.from_formula(formula=formula, data=model_data, family=sm.families.Binomial()).fit(disp=False)

    return glm.params[condition_key], glm.pvalues[condition_key]


def _run_deseq2(
    adata: AnnData,
    condition_key: str,
    comparisons: list[tuple[str, str]] | None = None,
    covariates: list[str] | None = None,
    layer: str | None = None,
    n_cpus: int = 10,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run DESeq2 analysis."""
    inference = DefaultInference(n_cpus=n_cpus)

    design = f"~ {condition_key}"
    if covariates:
        design += " + " + " + ".join(covariates)

    adata = adata.copy()
    adata.X = _get_layer(adata, layer).copy()
    adata.X = _to_dense(adata.X)

    # Run DESeq2
    with suppress_output(verbose):
        dds = DeseqDataSet(
            adata=adata,
            design=design,
            refit_cooks=True,
            inference=inference,
        )
        dds.deseq2()

        # Get results for each comparison
        results = []
        for treatment, ref in comparisons:
            stat_res = DeseqStats(
                dds,
                contrast=[condition_key, treatment, ref],
                inference=inference,
            )
            stat_res.summary()
            results_df = stat_res.results_df
            results_df["test_condition"] = treatment
            results_df["ref_condition"] = ref
            results.append(results_df)

    results_df = pd.concat(results)

    # Rename columns
    results_df["feature"] = results_df.index.values
    results_df.rename(
        columns={
            "pvalue": "pval",
            "log2FoldChange": "log2fc",
        },
        inplace=True,
    )

    return results_df[["feature", "test_condition", "ref_condition", "log2fc", "stat", "pval", "padj"]]


def _run_lrt_cuml(
    X: np.ndarray,
    model_data: pd.DataFrame,
    condition_key: str,
    covariates: list[str] | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Run likelihood ratio test using cuML's LogisticRegression.

    Parameters
    ----------
    X
        Feature vector to test
    model_data
        DataFrame containing condition and covariate values
    condition_key
        Name of condition column in model_data
    covariates
        Names of covariate columns in model_data
    verbose
        Whether to show warnings and errors

    Returns
    -------
    tuple
        Coefficient estimate and p-value from likelihood ratio test
    """
    try:
        from cuml.linear_model import LogisticRegression
    except ImportError as err:
        raise ImportError(
            "cuML is not installed. Please install it to use the 'lr_cuml' method for differential expression."
        ) from err

    y = model_data[condition_key].values

    # Prepare covariate matrix if provided
    if covariates:
        Z = model_data[covariates].values
        # Fit null model (intercept + covariates)
        X_null = np.column_stack([np.ones(X.shape[0]), Z])
        # Fit full model (intercept + feature + covariates)
        X_full = np.column_stack([np.ones_like(X), X, Z])
    else:
        # Fit null model (intercept only)
        X_null = np.ones((X.shape[0], 1))
        # Fit full model (intercept + feature)
        X_full = np.column_stack([np.ones_like(X), X])

    # Fit null model
    null_model = LogisticRegression(penalty="none")
    null_model.fit(X_null, y)
    null_prob = null_model.predict_proba(X_null)

    # Fit full model
    full_model = LogisticRegression(penalty="none")
    full_model.fit(X_full, y)
    full_prob = full_model.predict_proba(X_full)

    # Calculate log-likelihoods
    alt_log_likelihood = -log_loss(y, full_prob, normalize=False)
    null_log_likelihood = -log_loss(y, null_prob, normalize=False)

    # Likelihood ratio test
    lr_stat = 2 * (alt_log_likelihood - null_log_likelihood)
    # df = number of new parameters in full model (just the feature)
    pval = stats.chi2.sf(lr_stat, df=1)

    # Return feature coefficient (index 1 after intercept)
    return full_model.coef_.flatten()[1], pval


def _run_de(
    X: np.ndarray | sparse.spmatrix,
    model_data: pd.DataFrame,
    feature_names: pd.Index,
    method: str,
    condition_key: str,
    covariates: list[str] | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run GLM-based differential expression analysis.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix of shape (n_samples, n_features)
    model_data : pd.DataFrame
        DataFrame containing condition and covariate values
    feature_names : pd.Index
        Names of features/genes
    method : str
        DE method to use: "lr", "negbinom", "anova", "binomial"
    condition_key : str
        Name of condition column in model_data
    covariates : list[str]
        Names of covariate columns in model_data
    n_jobs : int
        Number of parallel jobs for running tests. Use -1 to use all processors.
    verbose : bool
        Whether to show progress and warnings
    """
    # Choose test function for non-batched methods
    if method == "lr":
        test_func = _run_lr_test
    elif method == "negbinom":
        test_func = _run_negbinom
    elif method == "anova" or method == "anova_residual":
        test_func = partial(_run_anova, method=method)
    elif method == "binomial":
        test_func = _run_binomial
    elif method == "lr_cuml":
        test_func = _run_lrt_cuml
    else:
        raise ValueError(f"Unsupported method: {method}")

    def _process_feature(i: int) -> tuple[str, float, float] | tuple[str, None, None]:
        """Process a single feature and return results or None if failed."""
        try:
            x = _to_dense(X[:, i]).flatten()
            coef, pval = test_func(x, model_data, condition_key, covariates, verbose=False)
            return feature_names[i], coef, pval
        except Exception as e:  # noqa: BLE001
            if verbose:
                warnings.warn(f"Error testing feature {feature_names[i]}: {str(e)}", stacklevel=2)
            return feature_names[i], None, None

    # Run tests in parallel with progress bar
    n_features = X.shape[1]

    # Process all features in parallel with a progress bar using joblib's generator return
    feature_results = list(
        tqdm.tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(delayed(_process_feature)(i) for i in range(n_features)),
            total=n_features,
            disable=not verbose,
        )
    )

    # Collect successful results and count errors
    results = {
        "feature": [],
        "coef": [],
        "pval": [],
    }
    errors = {}

    for feat, coef, pval in feature_results:
        if coef is not None and pval is not None:
            results["feature"].append(feat)
            results["coef"].append(coef)
            results["pval"].append(pval)
        else:
            errors[feat] = "Test failed"

    if len(errors) > 0 and verbose:
        warnings.warn(
            f"DE analysis failed for {len(errors)} features: {list(errors.keys())}",
            stacklevel=2,
        )

    results_df = pd.DataFrame(results)
    return results_df
