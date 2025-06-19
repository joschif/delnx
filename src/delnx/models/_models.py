"""Regression models in JAX."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy import optimize


@dataclass(frozen=True)
class Regression:
    """Base class for regression models.

    This is the abstract base class for all regression models in the package.
    It provides common functionality for fitting models, computing statistics,
    and handling offsets for normalization.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    tol : float, default=1e-6
        Convergence tolerance for optimization algorithms.
    optimizer : str, default="BFGS"
        Optimization method to use. Options include "BFGS" and "IRLS"
        (Iteratively Reweighted Least Squares) for GLM-type models.
    skip_stats : bool, default=False
        Whether to skip calculating Wald test statistics (for faster computation).
    offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
        Offset terms (on log scale for GLMs) to include in the model. Used for
        incorporating normalization factors like size factors in RNA-seq data.
    """

    maxiter: int = 100
    tol: float = 1e-6
    optimizer: str = "BFGS"
    skip_stats: bool = False
    offset: jnp.ndarray | None = None

    def _fit_bfgs(self, neg_ll_fn: Callable, init_params: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Fit model using the BFGS optimizer.

        Parameters
        ----------
        neg_ll_fn : Callable
            Function that computes the negative log-likelihood.
        init_params : jnp.ndarray
            Initial parameter values.
        **kwargs
            Additional arguments passed to the optimizer.

        Returns
        -------
        jnp.ndarray
            Optimized parameters.
        """
        result = optimize.minimize(neg_ll_fn, init_params, method="BFGS", options={"maxiter": self.maxiter})
        return result.x

    def _fit_irls(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        weight_fn: Callable,
        working_resid_fn: Callable,
        init_params: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Fit model using Iteratively Reweighted Least Squares algorithm.

        This implements the IRLS algorithm for generalized linear models
        with support for offset terms. For count models (e.g., Negative
        Binomial), the offset is used to incorporate size factors.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        weight_fn : Callable
            Function to compute weights at each iteration.
        working_resid_fn : Callable
            Function to compute working residuals at each iteration.
        init_params : jnp.ndarray
            Initial parameter values.
        offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
            Offset term (log scale for GLMs) to include in the model.
        **kwargs
            Additional arguments passed to weight_fn and working_resid_fn.

        Returns
        -------
        jnp.ndarray
            Optimized parameters.
        """
        n, p = X.shape
        eps = 1e-6

        # Handle offset
        if offset is None:
            offset = jnp.zeros(n)

        def irls_step(state):
            i, converged, beta = state

            # Compute weights and working residuals
            W = weight_fn(X, beta, offset=offset, **kwargs)
            z = working_resid_fn(X, y, beta, offset=offset, **kwargs)

            # Weighted design matrix
            W_sqrt = jnp.sqrt(W)
            X_weighted = X * W_sqrt[:, None]
            z_weighted = z * W_sqrt

            # Solve weighted least squares: (X^T W X) β = X^T W z
            XtWX = X_weighted.T @ X_weighted
            XtWz = X_weighted.T @ z_weighted
            beta_new = jax.scipy.linalg.solve(XtWX + eps * jnp.eye(p), XtWz, assume_a="pos")

            # Check convergence
            delta = jnp.max(jnp.abs(beta_new - beta))
            converged = delta < self.tol

            return i + 1, converged, beta_new

        def irls_cond(state):
            i, converged, _ = state
            return jnp.logical_and(i < self.maxiter, ~converged)

        # Initialize state
        state = (0, False, init_params)
        final_state = jax.lax.while_loop(irls_cond, irls_step, state)
        _, _, beta_final = final_state
        return beta_final

    def _compute_wald_test(
        self, neg_ll_fn: Callable, params: jnp.ndarray, test_idx: int = -1
    ) -> tuple[jnp.ndarray, float, float]:
        """Compute Wald test statistics for model coefficients.

        This method calculates standard errors, test statistics, and p-values
        for coefficient estimates using the Hessian of the negative log-likelihood.

        Parameters
        ----------
        neg_ll_fn : Callable
            Function that computes the negative log-likelihood.
        params : jnp.ndarray
            Parameter estimates.
        test_idx : int, default=-1
            Index of the parameter to test (default is the last parameter).

        Returns
        -------
        tuple[jnp.ndarray, float, float]
            Tuple containing (standard errors, test statistics, p-values).
        """
        hess_fn = jax.hessian(neg_ll_fn)
        hessian = hess_fn(params)
        hessian = 0.5 * (hessian + hessian.T)  # Ensure symmetry

        # Use pseudoinverse for better numerical stability
        cov = jnp.linalg.pinv(hessian)
        se = jnp.sqrt(jnp.clip(jnp.diag(cov), 1e-8))

        # Compute test statistic and p-value only if SE is valid
        stat = (params / se) ** 2
        pval = jsp.stats.chi2.sf(stat, df=1)

        return se, stat, pval

    def _exact_solution(self, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute exact Ordinary Least Squares solution.

        For linear regression, the offset is incorporated by adjusting the
        response variable (y - offset) rather than the linear predictor.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
            Offset term to include in the model.

        Returns
        -------
        jnp.ndarray
            Coefficient estimates.
        """
        if offset is not None:
            # Adjust y by subtracting offset for linear regression
            y_adj = y - offset
        else:
            y_adj = y

        XtX = X.T @ X
        Xty = X.T @ y_adj
        params = jax.scipy.linalg.solve(XtX, Xty, assume_a="pos")
        return params

    def get_llf(self, X: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray, offset: jnp.ndarray | None = None) -> float:
        """Get log-likelihood at fitted parameters.

        This method converts the negative log-likelihood to a log-likelihood
        value, which is useful for model comparison and likelihood ratio tests.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        params : jnp.ndarray
            Parameter estimates.
        offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
            Offset term to include in the model.

        Returns
        -------
        float
            Log-likelihood value.
        """
        nll = self._negative_log_likelihood(params, X, y, offset)
        return -nll  # Convert negative log-likelihood to log-likelihood


@dataclass(frozen=True)
class LinearRegression(Regression):
    """Linear regression with Ordinary Least Squares estimation.

    This class implements a basic linear regression model using OLS, with support for
    including offset terms. For linear models, offsets are applied by subtracting
    from the response variable rather than adding to the linear predictor.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization (inherited from Regression).
    tol : float, default=1e-6
        Convergence tolerance (inherited from Regression).
    optimizer : str, default="BFGS"
        Optimization method (inherited from Regression).
    skip_stats : bool, default=False
        Whether to skip calculating Wald test statistics (inherited from Regression).
    offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
        Offset terms to include in the model (inherited from Regression).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from delnx.models import LinearRegression
    >>> X = jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])  # Design matrix with intercept
    >>> y = jnp.array([1.0, 2.0, 3.0])  # Response variable
    >>> model = LinearRegression()
    >>> result = model.fit(X, y)
    >>> print(f"Coefficients: {result['coef']}")
    """

    def _negative_log_likelihood(
        self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> float:
        """Compute negative log likelihood (assuming Gaussian noise) with offset."""
        pred = jnp.dot(X, params)
        if offset is not None:
            pred = pred + offset
        residuals = y - pred
        return 0.5 * jnp.sum(residuals**2)

    def _compute_cov_matrix(
        self, X: jnp.ndarray, params: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute covariance matrix for parameters with offset."""
        n = X.shape[0]
        pred = X @ params
        if offset is not None:
            pred = pred + offset
        residuals = y - pred
        sigma2 = jnp.sum(residuals**2) / (n - len(params))
        return sigma2 * jnp.linalg.pinv(X.T @ X)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None) -> dict:
        """Fit linear regression model.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Response vector of shape (n_samples,).
        offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
            Offset term to include in the model. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        Dictionary containing:
                - coef: Parameter estimates
                - llf: Log-likelihood at fitted parameters
                - se: Standard errors (:obj:`None` if `skip_stats=True`)
                - stat: Test statistics (:obj:`None` if `skip_stats=True`)
                - pval: P-values (:obj:`None` if `skip_stats=True`)
        """
        # Fit model
        params = self._exact_solution(X, y, offset)

        # Compute standard errors
        llf = self.get_llf(X, y, params, offset)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_stats:
            cov = self._compute_cov_matrix(X, params, y, offset)
            se = jnp.sqrt(jnp.diag(cov))
            stat = (params[-1] / se[-1]) ** 2
            pval = jsp.stats.chi2.sf(stat, df=1)

        return {"coef": params, "llf": llf, "se": se, "stat": stat, "pval": pval}


@dataclass(frozen=True)
class LogisticRegression(Regression):
    """Logistic regression in JAX.

    This class implements logistic regression for binary classification tasks
    with support for offset terms. Offsets are added to the linear predictor
    before applying the logistic function.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    tol : float, default=1e-6
        Convergence tolerance for optimization algorithms.
    optimizer : str, default="BFGS"
        Optimization method to use. Options are "BFGS" or "IRLS" (recommended).
    skip_stats : bool, default=False
        Whether to skip calculating test statistics.
    offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
        Offset terms to include in the model.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from delnx.models import LogisticRegression
    >>> X = jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.5]])  # Design matrix with intercept
    >>> y = jnp.array([0.0, 0.0, 1.0])  # Binary outcome
    >>> model = LogisticRegression(optimizer="IRLS")
    >>> result = model.fit(X, y)
    >>> print(f"Coefficients: {result['coef']}")
    """

    def _negative_log_likelihood(
        self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> float:
        """Compute negative log likelihood with offset."""
        logits = jnp.dot(X, params)
        if offset is not None:
            logits = logits + offset
        nll = -jnp.sum(y * logits - jnp.logaddexp(0.0, logits))
        return nll

    def _weight_fn(self, X: jnp.ndarray, beta: jnp.ndarray, offset: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute weights for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        p = jax.nn.sigmoid(eta)
        return p * (1 - p)

    def _working_resid_fn(
        self, X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, offset: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute working residuals for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        p = jax.nn.sigmoid(eta)
        return eta + (y - p) / jnp.clip(p * (1 - p), 1e-6)

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> dict:
        """Fit logistic regression model.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Binary response vector of shape (n_samples,).
        offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
            Offset term to include in the model. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        Dictionary containing:
                - coef: Parameter estimates
                - llf: Log-likelihood at fitted parameters
                - se: Standard errors (:obj:`None` if `skip_stats=True`)
                - stat: Test statistics (:obj:`None` if `skip_stats=True`)
                - pval: P-values (:obj:`None` if `skip_stats=True`)
        """
        # Fit model
        init_params = jnp.zeros(X.shape[1])
        if self.optimizer == "BFGS":
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset)
            params = self._fit_bfgs(nll, init_params)
        elif self.optimizer == "IRLS":
            params = self._fit_irls(X, y, self._weight_fn, self._working_resid_fn, init_params, offset=offset)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Get log-likelihood
        llf = self.get_llf(X, y, params, offset)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_stats:
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset)
            se, stat, pval = self._compute_wald_test(nll, params)

        return {
            "coef": params,
            "llf": llf,
            "se": se,
            "stat": stat,
            "pval": pval,
        }


@dataclass(frozen=True)
class NegativeBinomialRegression(Regression):
    """Negative Binomial regression in JAX.

    This class implements Negative Binomial regression for modeling count data,
    particularly RNA-seq data, with support for offsets to incorporate size factors
    or other normalization terms. The model uses a log link function and allows for
    overdispersion in count data.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of iterations for optimization algorithms.
    tol : float, default=1e-6
        Convergence tolerance for optimization algorithms.
    optimizer : str, default="BFGS"
        Optimization method to use. Options are "BFGS" or "IRLS".
    skip_stats : bool, default=False
        Whether to skip calculating Wald test statistics.
    offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
        Offset terms (log scale) to include in the model. Typically log(size_factors)
        to account for differences in sequencing depth in RNA-seq analysis.
    dispersion : float | :obj:`None`, default=:obj:`None`
        Fixed dispersion parameter. If :obj:`None`, dispersion is estimated from the data.
    dispersion_range : tuple[float, float], default=(1e-6, 10.0)
        Range for valid dispersion values to prevent numerical issues.
    dispersion_method : str, default="moments"
        Method for estimating dispersion. Options are "moments" or "mle".

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from delnx.models import NegativeBinomialRegression
    >>> X = jnp.array([[1.0, 0.0], [1.0, 1.0]])  # Design matrix with intercept
    >>> y = jnp.array([10.0, 20.0])  # Count data
    >>> size_factors = jnp.array([0.8, 1.2])  # Size factors from normalization
    >>> offset = jnp.log(size_factors)  # Log transform for offset
    >>> model = NegativeBinomialRegression(optimizer="IRLS")
    >>> result = model.fit(X, y, offset=offset)
    >>> print(f"Coefficients: {result['coef']}")
    """

    dispersion: float | None = None
    dispersion_range: tuple[float, float] = (1e-6, 10.0)
    dispersion_method: str = "mle"

    def _negative_log_likelihood(
        self,
        params: jnp.ndarray,
        X: jnp.ndarray,
        y: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        dispersion: float = 1.0,
    ) -> float:
        """Compute negative log likelihood with offset."""
        eta = X @ params

        if offset is not None:
            eta = eta + offset

        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)

        # Get the size (r = alpha = 1 / dispersion)
        r = 1 / jnp.clip(dispersion, self.dispersion_range[0], self.dispersion_range[1])

        ll = (
            jsp.special.gammaln(r + y)
            - jsp.special.gammaln(r)
            - jsp.special.gammaln(y + 1)
            + r * jnp.log(r / (r + mu))
            + y * jnp.log(mu / (r + mu))
        )
        return -jnp.sum(ll)

    def _weight_fn(
        self, X: jnp.ndarray, beta: jnp.ndarray, offset: jnp.ndarray | None = None, dispersion: float = 1.0
    ) -> jnp.ndarray:
        """Compute weights for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)

        # Negative binomial variance = μ + φμ²
        var = mu + dispersion * mu**2
        # IRLS weights: (dμ/dη)² / var
        # For log link: dμ/dη = μ
        return mu**2 / jnp.clip(var, 1e-6)

    def _working_resid_fn(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        beta: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        dispersion: float = 1.0,
    ) -> jnp.ndarray:
        """Compute working residuals for IRLS with offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        eta = jnp.clip(eta, -50, 50)
        mu = jnp.exp(eta)

        # Working response: z = η + (y - μ) * (dη/dμ)
        # For log link: dη/dμ = 1/μ
        return eta + (y - mu) / mu

    def get_llf(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        params: jnp.ndarray,
        offset: jnp.ndarray | None = None,
        dispersion: float = 1.0,
    ) -> float:
        """Get log-likelihood at fitted parameters with offset."""
        nll = self._negative_log_likelihood(params, X, y, offset, dispersion)
        return -nll

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> dict:
        """Fit negative binomial regression model with optional offset.

        This method fits a Negative Binomial regression model to count data,
        with support for including offset terms (typically log size factors)
        to account for normalization. The method also handles dispersion
        estimation if not provided during initialization.

        Parameters
        ----------
        X : jnp.ndarray
            Design matrix of shape (n_samples, n_features).
        y : jnp.ndarray
            Count response vector of shape (n_samples,).
        offset : jnp.ndarray | :obj:`None`, default=:obj:`None`
            Offset term (log scale) to include in the model. Typically
            log(size_factors) for RNA-seq data. If provided, overrides
            the offset set during class initialization.

        Returns
        -------
        Dictionary containing:
                - coef: Parameter estimates
                - llf: Log-likelihood at fitted parameters
                - se: Standard errors (:obj:`None` if `skip_stats=True`)
                - stat: Test statistics (:obj:`None` if `skip_stats=True`)
                - pval: P-values (:obj:`None` if `skip_stats=True`)
                - dispersion: Estimated or provided dispersion parameter
        """
        # Estimate dispersion parameter
        if self.dispersion is not None:
            dispersion = jnp.clip(self.dispersion, self.dispersion_range[0], self.dispersion_range[1])
        else:
            dispersion = DispersionEstimator().estimate_dispersion_single_gene(y, self.dispersion_method)

        # Initialize parameters
        init_params = jnp.zeros(X.shape[1])

        # Better initialization for intercept
        mean_y = jnp.maximum(jnp.mean(y), 1e-8)
        if offset is not None:
            init_params = init_params.at[0].set(jnp.log(mean_y) - jnp.mean(offset))
        else:
            init_params = init_params.at[0].set(jnp.log(mean_y))

        # Fit model
        if self.optimizer == "BFGS":
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset, dispersion=dispersion)
            params = self._fit_bfgs(nll, init_params)
        elif self.optimizer == "IRLS":
            params = self._fit_irls(
                X, y, self._weight_fn, self._working_resid_fn, init_params, offset=offset, dispersion=dispersion
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Get log-likelihood
        llf = self.get_llf(X, y, params, offset, dispersion)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_stats:
            nll = partial(self._negative_log_likelihood, X=X, y=y, offset=offset, dispersion=dispersion)
            se, stat, pval = self._compute_wald_test(nll, params)

        return {
            "coef": params,
            "llf": llf,
            "se": se,
            "stat": stat,
            "pval": pval,
            "dispersion": dispersion,
        }


@dataclass(frozen=True)
class DispersionEstimator:
    """Estimate dispersion parameter for Negative Binomial regression.

    Parameters
    ----------
    dispersion_range : tuple[float, float]
        Range for valid dispersion values.
    shrinkage_weight_range : tuple[float, float]
        Range for shrinkage weights used in dispersion estimation.
    prior_variance : float
        Prior variance for Bayesian shrinkage methods.
    prior_df : float
        Prior degrees of freedom for empirical Bayes methods.
    """

    dispersion_range: tuple[float, float] = (1e-6, 10.0)
    shrinkage_weight_range: tuple[float, float] = (0.1, 0.95)
    prior_variance: float = 0.1
    prior_df: float = 5.0

    def estimate_dispersion_single_gene(
        self, x: jnp.ndarray, method: str = "mle", size_factors: jnp.ndarray | None = None
    ) -> float:
        """Estimate dispersion parameter for a single gene.

        Parameters
        ----------
        x : jnp.ndarray
            Raw expression counts for a single gene.
        method : str, optional
            Method to use for dispersion estimation:
                - "moments": Method of moments.
                - 'mle': Maximum likelihood estimation based an intercept-only model
        size_factors : jnp.ndarray, optional
            Size factors for normalization. If None, assumes all equal to 1.

        Returns
        -------
        Estimated dispersion parameter.
        """
        if size_factors is None:
            size_factors = jnp.ones_like(x)

        if method == "moments":
            return self._estimate_dispersion_moments(x, size_factors)
        elif method == "mle":
            return self._estimate_dispersion_mle(x, size_factors)
        else:
            raise ValueError(f"Unknown method for dispersion estimation: {method}")

    def estimate_dispersion(
        self,
        X: jnp.ndarray,
        method: str = "mle",
        size_factors: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Estimate gene-wise dispersion.

        Parameters
        ----------
        X : jnp.ndarray
            Raw expression counts for multiple genes, shape (n_samples, n_genes).
        method : str, optional
            Method to use for dispersion estimation:
                - "moments": Method of moments.
                - "mle": Maximum likelihood estimation.
        size_factors : jnp.ndarray, optional
            Size factors for each sample, shape (n_samples,). If None, assumes all equal to 1.

        Returns
        -------
        Estimated dispersion parameters for each gene.
        """
        if size_factors is None:
            size_factors = jnp.ones(X.shape[0])

        return jax.vmap(
            self.estimate_dispersion_single_gene,
            in_axes=(1, None, None),
        )(X, method, size_factors)

    @partial(jax.jit, static_argnums=(0,))
    def _estimate_dispersion_moments(self, x: jnp.ndarray, size_factors: jnp.ndarray | None = None) -> float:
        """Estimate dispersion parameter using method-of-moments on normalized counts."""
        if size_factors is None:
            size_factors = jnp.ones_like(x)

        # Work with normalized counts
        x_norm = x / size_factors
        # mean inverse size factor
        sf_mean_inv = (1 / size_factors).mean()

        # Estimate mean and variance of normalized counts
        mu_norm = jnp.mean(x_norm, axis=0)
        var_norm = jnp.var(x_norm, axis=0, ddof=1)

        # For negative binomial: Var = μ + φ * μ² -> φ = (Var - μ) / μ
        excess_var = var_norm - sf_mean_inv * mu_norm
        dispersion = jnp.nan_to_num(excess_var / (mu_norm**2))

        # Clip to valid range
        return jnp.clip(dispersion, self.dispersion_range[0], self.dispersion_range[1])

    @partial(jax.jit, static_argnums=(0,))
    def _estimate_dispersion_mle(self, x: jnp.ndarray, size_factors: jnp.ndarray | None = None) -> float:
        """Estimate dispersion parameter using maximum likelihood estimation."""
        if size_factors is None:
            size_factors = jnp.ones_like(x)

        # Get initial estimate from method of moments
        dispersion_init = self._estimate_dispersion_moments(x, size_factors)

        # Estimate base mean from normalized counts
        log_mu = jnp.log(jnp.maximum(jnp.mean(x / size_factors), 1e-6))
        offset = jnp.log(size_factors) if size_factors is not None else None

        def neg_ll(log_dispersion):
            # Get the size (r = alpha = 1 / dispersion)
            dispersion = jnp.exp(log_dispersion)
            r = 1 / jnp.clip(dispersion, self.dispersion_range[0], self.dispersion_range[1])

            eta = log_mu

            if offset is not None:
                eta = eta + offset

            eta = jnp.clip(eta, -50, 50)
            mu = jnp.exp(eta)

            ll = (
                jsp.special.gammaln(r + x)
                - jsp.special.gammaln(r)
                - jsp.special.gammaln(x + 1)
                + r * jnp.log(r / (r + mu))
                + x * jnp.log(mu / (r + mu))
            )
            return -jnp.sum(ll)

        # Initialize parameters on log scale for better optimization
        initial_params = jnp.array([jnp.log(dispersion_init)])
        result = optimize.minimize(neg_ll, initial_params, method="BFGS")

        # With lax to make jit compatible
        log_disp = jax.lax.cond(
            result.success,
            lambda x: x[1],
            lambda _: jnp.log(dispersion_init),
            result.x,
        )

        return jnp.clip(jnp.exp(log_disp), self.dispersion_range[0], self.dispersion_range[1])

    def shrink_dispersion(
        self, dispersions: jnp.ndarray, mu: jnp.ndarray, method: str = "deseq2", size_factors: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Fit a trend to the dispersion-mean relationship and shrink estimates.

        Parameters
        ----------
        dispersions : jnp.ndarray
            Gene-wise dispersion estimates.
        mu : jnp.ndarray
            Raw mean expression values for each gene.
        method : str, optional
            Shrinkage method to use:
                - "edger": Empirical Bayes shrinkage towards a log-linear trend.
                - "deseq2": Bayesian shrinkage towards a parametric trend.
        size_factors : jnp.ndarray, optional
            Size factors for normalization. Used to normalize mu for trend fitting.

        Returns
        -------
        Shrunk dispersion estimates.
        """
        # Normalize means for trend fitting
        if size_factors is not None:
            mu_normalized = mu / jnp.mean(size_factors)
        else:
            mu_normalized = mu

        # Ensure we have positive means for trend fitting
        mu_normalized = jnp.maximum(mu_normalized, 1e-6)

        if method == "edger":
            disp_trend = self._fit_trend_linear(dispersions, mu_normalized)
            return self._dispersion_shrinkage(dispersions, disp_trend, method="empirical_bayes")
        elif method == "deseq2":
            disp_trend = self._fit_trend_parametric(dispersions, mu_normalized)
            return self._dispersion_shrinkage(dispersions, disp_trend, method="bayesian")
        else:
            raise ValueError(f"Unknown method for dispersion shrinkage: {method}")

    def _fit_trend_linear(self, dispersions: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        """Fit linear trend to log(dispersion) vs log(mean)."""
        # Filter out extreme values for trend fitting
        valid_mask = (dispersions > self.dispersion_range[0]) & (dispersions < self.dispersion_range[1]) & (mu > 1e-6)

        if jnp.sum(valid_mask) < 10:
            # Not enough valid points, return median dispersion
            return jnp.full_like(dispersions, jnp.median(dispersions))

        valid_dispersions = dispersions[valid_mask]
        valid_mu = mu[valid_mask]

        # Fit linear trend on log scale: log(φ) = a + b * log(μ)
        log_means = jnp.log(valid_mu)
        log_disps = jnp.log(valid_dispersions)

        # Linear regression
        design = jnp.column_stack([jnp.ones_like(log_means), log_means])
        coefs = jnp.linalg.lstsq(design, log_disps, rcond=None)[0]

        # Predict for all genes
        all_log_means = jnp.log(mu)
        all_design = jnp.column_stack([jnp.ones_like(all_log_means), all_log_means])
        log_trend = all_design @ coefs
        trend = jnp.exp(log_trend)

        return jnp.clip(trend, self.dispersion_range[0], self.dispersion_range[1])

    def _fit_trend_parametric(self, dispersions: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        """Fit parametric trend: φ = a/μ + b (DESeq2-style)."""
        # Filter out extreme values for trend fitting
        valid_mask = (dispersions > self.dispersion_range[0]) & (dispersions < self.dispersion_range[1]) & (mu > 1e-6)

        if jnp.sum(valid_mask) < 10:
            return jnp.full_like(dispersions, jnp.median(dispersions))

        valid_dispersions = dispersions[valid_mask]
        valid_mu = mu[valid_mask]

        def gamma_trend(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            a, b = params
            return jnp.maximum(a / jnp.maximum(x, 1e-6) + b, 1e-6)

        def loss_fn(params: jnp.ndarray) -> float:
            predicted = gamma_trend(params, valid_mu)
            log_diff = jnp.log(valid_dispersions) - jnp.log(predicted)
            return jnp.sum(log_diff**2)

        # Initialize parameters
        mean_disp = jnp.mean(valid_dispersions)
        mean_mu = jnp.mean(valid_mu)
        initial_params = jnp.array(
            [
                mean_disp * mean_mu,  # a
                mean_disp * 0.1,  # b
            ]
        )

        result = optimize.minimize(loss_fn, initial_params, method="BFGS")

        # With lax to make jit compatible
        trend = jax.lax.cond(
            result.success,
            lambda x: gamma_trend(x, mu),
            lambda _: jnp.full_like(dispersions, jnp.median(dispersions)),
            result.x,
        )

        return jnp.clip(trend, self.dispersion_range[0], self.dispersion_range[1])

    @partial(jax.jit, static_argnums=(0, 3))
    def _dispersion_shrinkage(
        self,
        dispersions: jnp.ndarray,
        trend: jnp.ndarray,
        method: str = "empirical_bayes",
    ) -> jnp.ndarray:
        """Apply shrinkage to gene-wise dispersions towards trend."""
        log_genewise = jnp.log(jnp.maximum(dispersions, 1e-6))
        log_trend = jnp.log(jnp.maximum(trend, 1e-6))

        # Estimate the variability of gene-wise dispersions around trend
        log_diff = log_genewise - log_trend
        diff_var = jnp.maximum(jnp.var(log_diff, ddof=1), 0.01)

        if method == "empirical_bayes":
            # edgeR-style shrinkage
            shrinkage_weight = 1.0 / (self.prior_df * diff_var + 1.0)
        elif method == "bayesian":
            # DESeq2-style shrinkage
            shrinkage_weight = self.prior_variance / (self.prior_variance + diff_var)
        else:
            raise ValueError(f"Unknown shrinkage method: {method}")

        shrinkage_weight = jnp.clip(shrinkage_weight, self.shrinkage_weight_range[0], self.shrinkage_weight_range[1])

        # Shrink towards trend
        log_shrunk = shrinkage_weight * log_trend + (1 - shrinkage_weight) * log_genewise

        return jnp.clip(jnp.exp(log_shrunk), self.dispersion_range[0], self.dispersion_range[1])
