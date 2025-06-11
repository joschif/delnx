from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy import optimize as jsp_optimize


@dataclass(frozen=True)
class Regression:
    """Base class for regression models."""

    maxiter: int = 100
    tol: float = 1e-6
    optimizer: str = "BFGS"
    skip_wald: bool = False

    def _fit_bfgs(self, neg_ll_fn: Callable, init_params: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Fit using BFGS optimizer."""
        result = jsp_optimize.minimize(neg_ll_fn, init_params, method="BFGS", options={"maxiter": self.maxiter})
        return result.x

    def _fit_irls(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        weight_fn: Callable,
        working_resid_fn: Callable,
        **kwargs,
    ) -> jnp.ndarray:
        """Fit using IRLS algorithm."""
        n, p = X.shape
        eps = 1e-6

        # TODO: implement step size control
        def irls_step(state):
            i, converged, beta = state

            # Compute weights and working residuals
            W = weight_fn(X, beta, **kwargs)
            z = working_resid_fn(X, y, beta, **kwargs)

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

        # Inizialize intercept with log(mean(y))
        beta_init = jnp.zeros(p)
        beta_init = beta_init.at[0].set(jnp.log(jnp.mean(y) + 1e-8))
        state = (0, False, beta_init)
        final_state = jax.lax.while_loop(irls_cond, irls_step, state)
        _, _, beta_final = final_state
        return beta_final

    def _compute_wald_test(
        self, neg_ll_fn: Callable, params: jnp.ndarray, test_idx: int = -1
    ) -> tuple[jnp.ndarray, float, float]:
        """Compute Wald test for coefficients."""
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

    def _exact_solution(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute exact OLS solution."""
        XtX = X.T @ X
        Xty = X.T @ y
        params = jax.scipy.linalg.solve(XtX, Xty, assume_a="pos")
        return params

    def get_llf(self, X: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray) -> float:
        """Get log-likelihood at fitted parameters."""
        nll = self._negative_log_likelihood(params, X, y)
        return -nll  # Convert negative log-likelihood to log-likelihood


@dataclass(frozen=True)
class LinearRegression(Regression):
    """Linear regression with OLS."""

    def _negative_log_likelihood(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute negative log likelihood (assuming Gaussian noise)."""
        pred = jnp.dot(X, params)
        residuals = y - pred
        return 0.5 * jnp.sum(residuals**2)

    def _compute_cov_matrix(self, X: jnp.ndarray, params: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute covariance matrix for parameters."""
        n = X.shape[0]
        pred = X @ params
        residuals = y - pred
        sigma2 = jnp.sum(residuals**2) / (n - len(params))
        return sigma2 * jnp.linalg.pinv(X.T @ X)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> dict:
        """Fit linear regression model with optional ANOVA analysis."""
        # Fit model
        params = self._exact_solution(X, y)

        # Compute standard errors
        llf = self.get_llf(X, y, params)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_wald:
            cov = self._compute_cov_matrix(X, params, y)
            se = jnp.sqrt(jnp.diag(cov))
            stat = (params[-1] / se[-1]) ** 2
            pval = jsp.stats.chi2.sf(stat, df=1)

        return {"coef": params, "llf": llf, "se": se, "stat": stat, "pval": pval}


@dataclass(frozen=True)
class LogisticRegression(Regression):
    """Logistic regression with JAX."""

    def _negative_log_likelihood(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute negative log likelihood."""
        logits = jnp.dot(X, params)
        nll = -jnp.sum(y * logits - jnp.logaddexp(0.0, logits))
        return nll

    def _weight_fn(self, X: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        """Compute weights for IRLS."""
        eta = jnp.clip(X @ beta, -10, 10)
        p = jax.nn.sigmoid(eta)
        return p * (1 - p)

    def _working_resid_fn(self, X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        """Compute working residuals for IRLS."""
        eta = jnp.clip(X @ beta, -10, 10)
        p = jax.nn.sigmoid(eta)
        return X @ beta + (y - p) / jnp.clip(p * (1 - p), 1e-6)

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> dict:
        """Fit logistic regression model."""
        # Fit model
        if self.optimizer == "BFGS":
            nll = partial(self._negative_log_likelihood, X=X, y=y)
            params = self._fit_bfgs(nll, jnp.zeros(X.shape[1]))
        elif self.optimizer == "IRLS":  # irls
            params = self._fit_irls(X, y, self._weight_fn, self._working_resid_fn)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Get log-likelihood
        llf = self.get_llf(X, y, params)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_wald:
            nll = partial(self._negative_log_likelihood, X=X, y=y)
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
    """Negative Binomial regression with JAX."""

    dispersion: float | None = None
    dispersion_range: tuple[float, float] = (0.001, 10.0)
    estimation_method: str = "moments"

    def _negative_log_likelihood(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, dispersion: float) -> float:
        """Compute negative log likelihood."""
        eta = jnp.clip(X @ params, -10, 10)
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

    def _weight_fn(self, X: jnp.ndarray, beta: jnp.ndarray, dispersion: float) -> jnp.ndarray:
        """Compute weights for IRLS."""
        eta = jnp.clip(X @ beta, -50, 50)
        mu = jnp.exp(eta)
        # Negative binomial variance = μ + φμ²
        var = mu + dispersion * mu**2
        # IRLS weights: (dμ/dη)² / var
        # For log link: dμ/dη = μ
        return mu**2 / jnp.clip(var, 1e-6)

    def _working_resid_fn(self, X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, dispersion: float) -> jnp.ndarray:
        """Compute working residuals for IRLS."""
        eta = jnp.clip(X @ beta, -50, 50)
        mu = jnp.exp(eta)
        # Working response: z = η + (y - μ) * (dη/dμ)
        # For log link: dη/dμ = 1/μ
        return eta + (y - mu) / mu

    def get_llf(self, X: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray, dispersion: float) -> float:
        """Get log-likelihood at fitted parameters."""
        nll = self._negative_log_likelihood(params, X, y, dispersion)
        return -nll

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> dict:
        """Fit negative binomial regression model."""
        # Estimate dispersion parameter
        if self.dispersion is not None:
            dispersion = jnp.clip(self.dispersion, self.dispersion_range[0], self.dispersion_range[1])
        else:
            dispersion = DispersionEstimator.estimate_dispersion_single_gene(y, self.estimation_method)

        # Fit model
        if self.optimizer == "BFGS":
            nll = partial(self._negative_log_likelihood, X=X, y=y, dispersion=dispersion)
            params = self._fit_bfgs(nll, jnp.zeros(X.shape[1]))
        elif self.optimizer == "IRLS":
            params = self._fit_irls(X, y, self._weight_fn, self._working_resid_fn, dispersion=dispersion)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Get log-likelihood
        llf = self.get_llf(X, y, params, dispersion)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_wald:
            nll = partial(self._negative_log_likelihood, X=X, y=y, dispersion=dispersion)
            se, stat, pval = self._compute_wald_test(nll, params)

        return {
            "coef": params,
            "llf": llf,
            "se": se,
            "stat": stat,
            "pval": pval,
        }


@dataclass(frozen=True)
class DispersionEstimator:
    """Estimate dispersion parameter for Negative Binomial regression."""

    method: str = "moments"
    dispersion_range: tuple[float, float] = (0.001, 10.0)
    prior_df: float = 10.0

    def estimate_dispersion_single_gene(self, y: jnp.ndarray) -> float:
        """Estimate dispersion parameter for a single gene."""
        if self.method == "moments":
            return self._estimate_dispersion_moments(y)
        elif self.method == "intercept":
            return self._estimate_dispersion_intercept(y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _estimate_dispersion_moments(self, y: jnp.ndarray) -> float:
        """Estimate dispersion parameter for a single gene using method of moments."""
        mu = jnp.clip(jnp.mean(y), 1e-6)
        var = jnp.clip(jnp.var(y), 1e-6)
        # Variance = μ + φμ²
        # φ = (s² - μ)/μ²
        dispersion = jnp.clip((var - mu) / (mu**2), self.dispersion_range[0], self.dispersion_range[1])
        return dispersion, mu, var

    def _estimate_dispersion_intercept(self, y: jnp.ndarray) -> float:
        """Estimate dispersion parameter for a single gene using intercept-only model."""
        # Estimate initial dispersion using method of moments
        dispersion_init, mu, _ = self._estimate_dispersion_moments(y)

        def neg_ll(log_dispersion):
            # Get the size (r = alpha = 1 / dispersion)
            dispersion = jnp.clip(jnp.exp(log_dispersion), self.dispersion_range[0], self.dispersion_range[1])
            r = 1 / dispersion
            ll = (
                jsp.special.gammaln(r + y)
                - jsp.special.gammaln(r)
                - jsp.special.gammaln(y + 1)
                + r * jnp.log(r / (r + mu))
                + y * jnp.log(mu / (r + mu))
            )
            return -jnp.sum(ll)

        # Optimize log(dispersion)
        log_dispersion_init = jnp.log(jnp.array([dispersion_init]))
        result = jax.scipy.optimize.minimize(neg_ll, log_dispersion_init, method="BFGS")
        return jnp.clip(jnp.exp(result.x[0]), 0.01, 10.0)

    def estimate_dispersion(
        self, Y: jnp.ndarray, size_factors: jnp.ndarray | None = None, method: str = "edger"
    ) -> jnp.ndarray:
        """Estimate gene-wise dispersion parameters."""
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _estimate_genewise_dispersion(self, Y: jnp.ndarray, size_factors: jnp.ndarray) -> float:
        """Estimate dispersion for a single gene using method of moments."""
        # Method of moments estimate
        normalized_Y = Y / size_factors
        mu = jnp.clip(jnp.mean(normalized_Y, axis=0), 1e-6)
        var = jnp.var(normalized_Y, axis=0, ddof=1)

        # Method of moments: dispersion = (var - mean) / mean^2
        dispersion = (var - mu) / (mu**2)
        dispersion = jnp.clip(dispersion, self.dispersion_range[0], self.dispersion_range[1])

        return dispersion

    def _fit_trend(self, dispersions: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        """Fit smooth trend to dispersions using local regression."""
        # Filter out extreme values for trend fitting
        valid_mask = (dispersions > self.dispersion_range[0]) & (dispersions < self.dispersion_range[1]) & (mu > 1.0)

        if jnp.sum(valid_mask) < 10:
            return jnp.full_like(dispersions, jnp.median(dispersions))

        valid_dispersions = dispersions[valid_mask]
        valid_mu = mu[valid_mask]

        # Use log scale for more stable fitting
        log_means = jnp.log(valid_mu)
        log_disps = jnp.log(valid_dispersions)

        # Fit simple linear trend on log scale
        design = jnp.column_stack([jnp.ones_like(log_means), log_means])
        coefs = jnp.linalg.lstsq(design, log_disps, rcond=None)[0]

        # Predict for all genes
        all_log_means = jnp.log(jnp.maximum(mu, 1.0))
        all_design = jnp.column_stack([jnp.ones_like(all_log_means), all_log_means])
        trend = jnp.exp(all_design @ coefs)

        return jnp.clip(trend, self.dispersion_range[0], self.dispersion_range[1])

    @partial(jax.jit, static_argnums=(0,))
    def _shrink_dispersions(self, dispersions: jnp.ndarray, trend: jnp.ndarray) -> jnp.ndarray:
        """Apply empirical Bayes shrinkage toward trend."""
        log_genewise = jnp.log(jnp.maximum(dispersions, 1e-6))
        log_trend = jnp.log(jnp.maximum(trend, 1e-6))

        # Estimate the variability of gene-wise dispersions
        log_diff = log_genewise - log_trend
        residual_var = jnp.maximum(jnp.var(log_diff), 0.1)

        # Shrinkage intensity based on residual variance and prior df
        shrinkage_weight = self.prior_df / (self.prior_df + 1.0 / residual_var)
        shrinkage_weight = jnp.clip(shrinkage_weight, 0.1, 0.9)

        log_shrunk = shrinkage_weight * log_trend + (1 - shrinkage_weight) * log_genewise
        return jnp.clip(jnp.exp(log_shrunk), 1e-6, 10.0)
