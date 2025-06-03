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

    alpha: float | None = None
    alpha_range: tuple[float, float] = (0.01, 10.0)
    estimation_method: str = "moments"

    def _estimate_alpha(self, y: jnp.ndarray) -> float:
        """Estimate dispersion parameter using specified method."""
        if self.alpha is not None:
            return self.alpha

        if self.estimation_method not in ["moments", "intercept"]:
            raise ValueError(
                f"Unknown estimation_method: {self.estimation_method}. Must be either 'moments' or 'intercept'."
            )

        if self.estimation_method == "intercept":
            return self._estimate_alpha_intercept(y)
        else:  # method of moments (default)
            return self._estimate_alpha_moments(y)[0]

    def _estimate_alpha_moments(self, y: jnp.ndarray) -> float:
        """Estimate dispersion parameter using method of moments."""
        mu = jnp.clip(jnp.mean(y), 1e-6)
        var = jnp.clip(jnp.var(y), 1e-6)
        alpha = jnp.clip((var - mu) / (mu**2), self.alpha_range[0], self.alpha_range[1])
        return alpha, mu, var

    def _estimate_alpha_intercept(self, y: jnp.ndarray) -> float:
        """Estimate dispersion parameter using intercept-only model."""
        # Estimate initial alpha using method of moments
        alpha_init, mu, _ = self._estimate_alpha_moments(y)

        def neg_ll(log_alpha):
            alpha = jnp.clip(jnp.exp(log_alpha), 0.01, 10.0)
            r = 1.0 / alpha
            ll = (
                jsp.special.gammaln(r + y)
                - jsp.special.gammaln(r)
                - jsp.special.gammaln(y + 1)
                + r * jnp.log(r / (r + mu))
                + y * jnp.log(mu / (r + mu))
            )
            return -jnp.sum(ll)

        # Optimize log(alpha)
        log_alpha_init = jnp.log(jnp.array([alpha_init]))
        result = jax.scipy.optimize.minimize(neg_ll, log_alpha_init, method="BFGS")
        return jnp.clip(jnp.exp(result.x[0]), 0.01, 10.0)

    def _negative_log_likelihood(self, params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, alpha: float) -> float:
        """Compute negative log likelihood."""
        eta = jnp.clip(X @ params, -10, 10)
        mu = jnp.exp(eta)
        alpha = jnp.clip(alpha, self.alpha_range[0], self.alpha_range[1])
        r = 1.0 / alpha

        ll = (
            jsp.special.gammaln(r + y)
            - jsp.special.gammaln(r)
            - jsp.special.gammaln(y + 1)
            + r * jnp.log(r / (r + mu))
            + y * jnp.log(mu / (r + mu))
        )
        return -jnp.sum(ll)

    def _weight_fn(self, X: jnp.ndarray, beta: jnp.ndarray, alpha: float) -> jnp.ndarray:
        """Compute weights for IRLS."""
        eta = jnp.clip(X @ beta, -50, 50)
        mu = jnp.exp(eta)
        var = mu + alpha * mu**2
        return 1 / jnp.clip(var, 1e-6)

    def _working_resid_fn(self, X: jnp.ndarray, y: jnp.ndarray, beta: jnp.ndarray, alpha: float) -> jnp.ndarray:
        """Compute working residuals for IRLS."""
        eta = jnp.clip(X @ beta, -50, 50)
        mu = jnp.exp(eta)
        return eta + (y / jnp.clip(mu, 1e-6) - 1)

    def get_llf(self, X: jnp.ndarray, y: jnp.ndarray, params: jnp.ndarray, alpha: float) -> float:
        """Get log-likelihood at fitted parameters."""
        nll = self._negative_log_likelihood(params, X, y, alpha)
        return -nll

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> dict:
        """Fit negative binomial regression model."""
        # Estimate dispersion parameter
        alpha = self._estimate_alpha(y)

        # Fit model
        if self.optimizer == "BFGS":
            nll = partial(self._negative_log_likelihood, X=X, y=y, alpha=alpha)
            params = self._fit_bfgs(nll, jnp.zeros(X.shape[1]))
        elif self.optimizer == "IRLS":
            params = self._fit_irls(X, y, self._weight_fn, self._working_resid_fn, alpha=alpha)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        # Get log-likelihood
        llf = self.get_llf(X, y, params, alpha)

        # Compute test statistics if requested
        se = stat = pval = None
        if not self.skip_wald:
            nll = partial(self._negative_log_likelihood, X=X, y=y, alpha=alpha)
            se, stat, pval = self._compute_wald_test(nll, params)

        return {
            "coef": params,
            "llf": llf,
            "se": se,
            "stat": stat,
            "pval": pval,
        }
