"""JAX implementations of PyDESeq2 functions for negative binomial log-likelihood and dispersion fitting."""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def nb_nll(counts: jnp.ndarray, mu: jnp.ndarray, alpha: float | jnp.ndarray) -> float | jnp.ndarray:
    r"""Neg log-likelihood of a negative binomial of parameters ``mu`` and ``alpha``.

    Mathematically, if ``counts`` is a vector of counting entries :math:`y_i`
    then the likelihood of each entry :math:`y_i` to be drawn from a negative
    binomial :math:`NB(\mu, \alpha)` is [1]

    .. math::
        p(y_i | \mu, \alpha) = \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        }
        \left(\frac{1}{1 + \alpha \mu} \right)^{1/\alpha}
        \left(\frac{\mu}{\alpha^{-1} + \mu} \right)^{y_i}

    As a consequence, assuming there are :math:`n` entries,
    the total negative log-likelihood for ``counts`` is

    .. math::
        \ell(\mu, \alpha) = \frac{n}{\alpha} \log(\alpha) +
            \sum_i \left \lbrace
            - \log \left( \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        } \right)
        + (\alpha^{-1} + y_i) \log (\alpha^{-1} + \mu)
        - y_i \log \mu
            \right \rbrace

    This is implemented in this function.

    Parameters
    ----------
    counts : ndarray
        Observations.

    mu : ndarray
        Mean of the distribution :math:`\mu`.

    alpha : float or ndarray
        Dispersion of the distribution :math:`\alpha`,
        s.t. the variance is :math:`\mu + \alpha \mu^2`.

    Returns
    -------
    float or ndarray
        Negative log likelihood of the observations counts
        following :math:`NB(\mu, \alpha)`.

    Notes
    -----
    [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """
    n = counts.shape[0]
    alpha = jnp.clip(alpha, 1e-10, 1e10)
    mu = jnp.clip(mu, 1e-10, 1e10)
    alpha_inv = 1.0 / alpha

    # Compute log-binomial coefficient using gammaln for numerical stability
    epsilon = 0.0
    logbinom = gammaln(counts + alpha_inv + epsilon) - gammaln(counts + 1 + epsilon) - gammaln(alpha_inv + epsilon)

    # Robust logarithm calculations
    log_terms = (counts + alpha_inv) * jnp.log(jnp.clip(alpha_inv + mu, 1e-15, None)) - counts * jnp.log(mu)

    # Handle potential NaN/inf values
    logbinom = jnp.where(jnp.isfinite(logbinom), logbinom, 0.0)
    log_terms = jnp.where(jnp.isfinite(log_terms), log_terms, 0.0)

    # Final NLL computation
    nll = n * alpha_inv * jnp.log(alpha) - jnp.sum(logbinom) + jnp.sum(log_terms)

    return nll


# Vectorize over alpha parameter
nb_nll_vmap = jax.vmap(nb_nll, in_axes=(None, None, 0))


def safe_slogdet(matrix):
    """Robust slogdet computation for JAX with regularization"""
    eye = jnp.eye(matrix.shape[-1])
    reg_matrix = matrix + 1e-12 * eye

    sign, logdet = jnp.linalg.slogdet(reg_matrix)

    logdet = jnp.where(jnp.isfinite(logdet), logdet, -1e10)
    sign = jnp.where(jnp.isfinite(sign), sign, 0.0)

    return sign, logdet


@partial(jax.jit, static_argnames=("grid_length", "cr_reg", "prior_reg"))
def grid_fit_alpha(
    counts: jnp.ndarray,
    design_matrix: jnp.ndarray,
    mu: jnp.ndarray,
    alpha_hat: float,
    min_disp: float,
    max_disp: float,
    prior_disp_var: float,
    cr_reg: bool,
    prior_reg: bool,
    grid_length: int = 100,
) -> float:
    """Find best dispersion parameter.

    Performs 1D grid search to maximize negative binomial log-likelihood.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    design_matrix : ndarray
        Design matrix.

    mu : ndarray
        Mean estimation for the NB model.

    alpha_hat : float
        Initial dispersion estimate.

    min_disp : float
        Lower threshold for dispersion parameters.

    max_disp : float
        Upper threshold for dispersion parameters.

    prior_disp_var : float, optional
        Prior dispersion variance.

    cr_reg : bool
        Whether to use Cox-Reid regularization. (default: ``True``).

    prior_reg : bool
        Whether to use prior log-residual regularization. (default: ``False``).

    grid_length : int
        Number of grid points. (default: ``100``).

    Returns
    -------
    float
        Logarithm of the fitted dispersion parameter.
    """
    min_log_alpha = jnp.log(min_disp)
    max_log_alpha = jnp.log(max_disp)
    grid = jnp.linspace(min_log_alpha, max_log_alpha, grid_length)

    def loss(log_alpha: jnp.ndarray) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        alpha = jnp.clip(alpha, min_disp, max_disp)

        W = mu[:, None] / (1 + mu[:, None] * alpha)
        W = jnp.clip(W, 1e-15, 1e6)

        base_loss = nb_nll_vmap(counts, mu, alpha)

        cr_term = jnp.where(
            cr_reg,
            0.5 * safe_slogdet((design_matrix.T[:, :, None] * W).transpose(2, 0, 1) @ design_matrix)[1],
            0.0,
        )

        prior_term = jnp.where(
            prior_reg,
            (jnp.log(alpha) - jnp.log(alpha_hat)) ** 2 / (2 * prior_disp_var),
            0.0,
        )

        total_loss = base_loss + cr_term + prior_term

        return jnp.where(jnp.isfinite(total_loss), total_loss, 1e10)

    ll_grid = loss(grid)

    min_idx = jnp.argmin(ll_grid)
    delta = grid[1] - grid[0]
    fine_grid = jnp.linspace(grid[min_idx] - delta, grid[min_idx] + delta, grid_length)

    ll_grid = loss(fine_grid)

    min_idx = jnp.argmin(ll_grid)
    log_alpha = fine_grid[min_idx]
    return log_alpha
