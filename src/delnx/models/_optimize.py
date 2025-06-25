"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm with bounds support."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search


class _BFGSResults(NamedTuple):
    """Results from BFGS optimization.

    Parameters
    ----------
      converged: True if minimization converged.
      failed: True if line search failed.
      k: integer the number of iterations of the BFGS update.
      nfev: integer total number of objective evaluations performed.
      ngev: integer total number of jacobian evaluations
      nhev: integer total number of hessian evaluations
      x_k: array containing the last argument value found during the search. If
        the search converged, then this value is the argmin of the objective
        function.
      f_k: array containing the value of the objective function at `x_k`. If the
        search converged, then this is the (local) minimum of the objective
        function.
      g_k: array containing the gradient of the objective function at `x_k`. If
        the search converged the l2-norm of this tensor should be below the
        tolerance.
      H_k: array containing the inverse of the estimated Hessian.
      status: int describing end state.
      line_search_status: int describing line search end state (only means
        something if line search fails).
    """

    converged: bool | jax.Array
    failed: bool | jax.Array
    k: int | jax.Array
    nfev: int | jax.Array
    ngev: int | jax.Array
    nhev: int | jax.Array
    x_k: jax.Array
    f_k: jax.Array
    g_k: jax.Array
    H_k: jax.Array
    old_old_fval: jax.Array
    status: int | jax.Array
    line_search_status: int | jax.Array


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def _project_bounds(x: jax.Array, bounds: tuple[jax.Array, jax.Array] | None) -> jax.Array:
    """Project x onto the bounds constraints."""
    if bounds is None:
        return x
    lower, upper = bounds
    return jnp.clip(x, lower, upper)


def _project_gradient(g: jax.Array, x: jax.Array, bounds: tuple[jax.Array, jax.Array] | None) -> jax.Array:
    """Project gradient for bound-constrained optimization.

    Sets gradient components to zero for variables at bounds where the gradient
    would push further outside the feasible region.
    """
    if bounds is None:
        return g

    lower, upper = bounds
    # For variables at lower bound with negative gradient, set gradient to 0
    at_lower = (x <= lower) & (g < 0)
    # For variables at upper bound with positive gradient, set gradient to 0
    at_upper = (x >= upper) & (g > 0)

    projected_g = jnp.where(at_lower | at_upper, 0.0, g)
    return projected_g


def _validate_bounds(bounds: tuple[jax.Array, jax.Array] | None, x0: jax.Array) -> tuple[jax.Array, jax.Array] | None:
    """Validate and process bounds."""
    if bounds is None:
        return None

    lower, upper = bounds

    # Convert to arrays with same shape as x0
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)

    # Broadcast to x0 shape if needed
    if lower.shape != x0.shape:
        if lower.ndim == 0:
            lower = jnp.full_like(x0, lower)
        else:
            lower = jnp.broadcast_to(lower, x0.shape)

    if upper.shape != x0.shape:
        if upper.ndim == 0:
            upper = jnp.full_like(x0, upper)
        else:
            upper = jnp.broadcast_to(upper, x0.shape)

    # Replace infinite bounds
    lower = jnp.where(jnp.isfinite(lower), lower, -jnp.inf)
    upper = jnp.where(jnp.isfinite(upper), upper, jnp.inf)

    return (lower, upper)


def minimize_bfgs(
    fun: Callable,
    x0: jax.Array,
    jac: Callable | None = None,
    bounds: tuple[jax.Array, jax.Array] | None = None,
    maxiter: int | None = None,
    norm=jnp.inf,
    gtol: float = 1e-5,
    line_search_maxiter: int = 10,
) -> _BFGSResults:
    """Minimize a function using BFGS with optional bounds constraints.

    Implements the BFGS algorithm from
      Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg.
      136-143.

    For bound-constrained problems, uses a projected gradient approach where:
    - Variables are projected onto bounds after each step
    - Gradients are projected to handle active constraints

    Args:
      fun: function of the form f(x) where x is a flat ndarray and returns a real
        scalar. The function should be composed of operations with vjp defined.
      x0: initial guess.
      jac: function that computes the gradient of fun. If None, uses automatic
        differentiation via jax.grad(fun).
      bounds: tuple of (lower, upper) bounds for each variable. Each bound can be
        a scalar (applied to all variables) or array with same shape as x0.
        Use -jnp.inf/jnp.inf for unbounded variables.
      maxiter: maximum number of iterations.
      norm: order of norm for convergence check. Default inf.
      gtol: terminates minimization when |grad|_norm < g_tol.
      line_search_maxiter: maximum number of linesearch iterations.

    Returns
    -------
      Optimization result.
    """
    x0 = jnp.asarray(x0)

    # Validate and process bounds
    bounds = _validate_bounds(bounds, x0)

    # Project initial guess onto bounds
    x0 = _project_bounds(x0, bounds)

    if maxiter is None:
        maxiter = jnp.size(x0) * 200

    d = x0.shape[0]

    # Set up gradient function
    if jac is None:

        def value_and_grad_fun(x):
            return jax.value_and_grad(fun)(x)

    else:

        def value_and_grad_fun(x):
            return fun(x), jac(x)

    initial_H = jnp.eye(d, dtype=x0.dtype)
    f_0, g_0 = value_and_grad_fun(x0)

    # Project gradient for bound constraints
    g_0_proj = _project_gradient(g_0, x0, bounds)

    state = _BFGSResults(
        converged=jnp.linalg.norm(g_0_proj, ord=norm) < gtol,
        failed=False,
        k=0,
        nfev=1,
        ngev=1,
        nhev=0,
        x_k=x0,
        f_k=f_0,
        g_k=g_0,
        H_k=initial_H,
        old_old_fval=f_0 + jnp.linalg.norm(g_0) / 2,
        status=0,
        line_search_status=0,
    )

    def cond_fun(state):
        return jnp.logical_not(state.converged) & jnp.logical_not(state.failed) & (state.k < maxiter)

    def body_fun(state):
        # Use projected gradient for search direction
        g_proj = _project_gradient(state.g_k, state.x_k, bounds)
        p_k = -_dot(state.H_k, g_proj)

        # For bounded problems, we need to be more careful with line search
        if bounds is not None:
            # Define wrapper functions that handle bounds
            def bounded_fun_wrapper(x):
                x_proj = _project_bounds(x, bounds)
                return fun(x_proj)

            # Use original gradient function for line search
            if jac is None:
                bounded_grad_wrapper = jax.grad(bounded_fun_wrapper)
            else:

                def bounded_grad_wrapper(x):
                    x_proj = _project_bounds(x, bounds)
                    return jac(x_proj)

            line_search_fun = bounded_fun_wrapper
        else:
            line_search_fun = fun

        # Perform line search
        line_search_results = line_search(
            line_search_fun,
            state.x_k,
            p_k,
            old_fval=state.f_k,
            old_old_fval=state.old_old_fval,
            gfk=g_proj,
            maxiter=line_search_maxiter,
        )

        # Handle line search failure with fallback strategy
        def try_fallback_step():
            # Try a conservative gradient step if line search fails
            grad_norm = jnp.linalg.norm(g_proj) + 1e-8
            alpha_fallback = jnp.minimum(0.01 / grad_norm, 1.0)
            s_k_fallback = alpha_fallback * p_k
            x_test = state.x_k + s_k_fallback
            if bounds is not None:
                x_test = _project_bounds(x_test, bounds)

            f_test, g_test = value_and_grad_fun(x_test)

            # Create fallback line search result with all required fields
            fallback_result = line_search_results._replace(
                failed=f_test >= state.f_k,  # Still failed if no improvement
                a_k=alpha_fallback,
                f_k=f_test,
                g_k=g_test,
                nfev=line_search_results.nfev + 1,
                ngev=line_search_results.ngev + 1,
                status=jnp.where(f_test < state.f_k, 0, 4),
            )
            return fallback_result

        def use_line_search_result():
            return line_search_results

        # Use fallback if line search failed
        final_line_search = lax.cond(line_search_results.failed, try_fallback_step, use_line_search_result)

        state = state._replace(
            nfev=state.nfev + final_line_search.nfev,
            ngev=state.ngev + final_line_search.ngev,
            failed=final_line_search.failed,
            line_search_status=final_line_search.status,
        )

        # Update position and project onto bounds if needed
        s_k = final_line_search.a_k * p_k
        x_kp1 = state.x_k + s_k
        if bounds is not None:
            x_kp1 = _project_bounds(x_kp1, bounds)

        # Get function and gradient values from line search result
        f_kp1 = final_line_search.f_k
        g_kp1 = final_line_search.g_k

        # BFGS update using actual step taken
        s_k_actual = x_kp1 - state.x_k
        y_k = g_kp1 - state.g_k

        # Skip BFGS update if curvature condition is not satisfied
        rho_k_denom = _dot(y_k, s_k_actual)

        # Only do BFGS update if we have positive curvature
        def bfgs_update():
            rho_k = jnp.reciprocal(rho_k_denom)
            sy_k = s_k_actual[:, jnp.newaxis] * y_k[jnp.newaxis, :]
            w = jnp.eye(d, dtype=rho_k.dtype) - rho_k * sy_k
            H_new = (
                _einsum("ij,jk,lk", w, state.H_k, w) + rho_k * s_k_actual[:, jnp.newaxis] * s_k_actual[jnp.newaxis, :]
            )
            return H_new

        def keep_old_hessian():
            return state.H_k

        # Update Hessian only if curvature condition is satisfied
        H_kp1 = lax.cond(rho_k_denom > 1e-10, bfgs_update, keep_old_hessian)

        # Check convergence using projected gradient
        g_kp1_proj = _project_gradient(g_kp1, x_kp1, bounds)
        converged = jnp.linalg.norm(g_kp1_proj, ord=norm) < gtol

        state = state._replace(
            converged=converged,
            k=state.k + 1,
            x_k=x_kp1,
            f_k=f_kp1,
            g_k=g_kp1,
            H_k=H_kp1,
            old_old_fval=state.f_k,
        )
        return state

    state = lax.while_loop(cond_fun, body_fun, state)

    # Final convergence check with projected gradient
    g_final_proj = _project_gradient(state.g_k, state.x_k, bounds)
    final_converged = jnp.linalg.norm(g_final_proj, ord=norm) < gtol

    status = jnp.where(
        final_converged,
        0,  # converged
        jnp.where(
            state.k == maxiter,
            1,  # max iters reached
            jnp.where(
                state.failed,
                2 + state.line_search_status,  # ls failed (+ reason)
                -1,  # undefined
            ),
        ),
    )

    state = state._replace(converged=final_converged, status=status)
    return state


class OptimizeResults(NamedTuple):
    """Object holding optimization results.

    Parameters
    ----------
      x: final solution.
      success: ``True`` if optimization succeeded.
      status: integer solver specific return code. 0 means converged (nominal),
        1=max BFGS iters reached, 3=zoom failed, 4=saddle point reached,
        5=max line search iters reached, -1=undefined
      fun: final function value.
      jac: final jacobian array.
      hess_inv: final inverse Hessian estimate.
      nfev: integer number of function calls used.
      njev: integer number of gradient evaluations.
      nit: integer number of iterations of the optimization algorithm.
    """

    x: jax.Array
    success: bool | jax.Array
    status: int | jax.Array
    fun: jax.Array
    jac: jax.Array
    hess_inv: jax.Array | None
    nfev: int | jax.Array
    njev: int | jax.Array
    nit: int | jax.Array


def minimize(
    fun: Callable,
    x0: jax.Array,
    args: tuple = (),
    *,
    method: str = "BFGS",
    jac: Callable | None = None,
    bounds: tuple[jax.Array, jax.Array] | None = None,
    tol: float | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResults:
    """Minimization of scalar function of one or more variables.

    This API matches SciPy with enhancements for JAX:

    - Gradients of ``fun`` are calculated automatically using JAX's autodiff
      support when ``jac`` is not provided.
    - Support for custom Jacobian functions via the ``jac`` parameter.
    - Support for box constraints via the ``bounds`` parameter.
    - The ``method`` argument defaults to "BFGS".
    - Optimization results may differ from SciPy due to differences in the line
      search implementation.

    ``minimize`` supports :func:`~jax.jit` compilation.

    Args:
      fun: the objective function to be minimized, ``fun(x, *args) -> float``,
        where ``x`` is a 1-D array with shape ``(n,)`` and ``args`` is a tuple
        of the fixed parameters needed to completely specify the function.
        ``fun`` must support differentiation if ``jac`` is not provided.
      x0: initial guess. Array of real elements of size ``(n,)``, where ``n`` is
        the number of independent variables.
      args: extra arguments passed to the objective function and Jacobian.
      method: solver type. Currently only ``"BFGS"`` is supported.
      jac: method for computing the gradient vector. If None (default), the
        gradient is computed using JAX's automatic differentiation. If callable,
        it should return the gradient vector ``jac(x, *args) -> array_like``.
      bounds: bounds for variables as a tuple ``(lower, upper)`` where each
        bound is an array with the same shape as ``x0`` or a scalar (applied
        to all variables). Use ``-jnp.inf`` and ``jnp.inf`` to indicate no bound.
      tol: tolerance for termination. For detailed control, use solver-specific
        options.
      options: a dictionary of solver options. All methods accept the following
        generic options:

        - maxiter (int): Maximum number of iterations to perform.
        - gtol (float): Gradient norm tolerance for convergence (default 1e-5).
        - norm: Order of norm for convergence check (default jnp.inf).
        - line_search_maxiter (int): Maximum line search iterations (default 10).

    Returns
    -------
      An :class:`OptimizeResults` object.
    """
    if options is None:
        options = {}

    if not isinstance(args, tuple):
        msg = "args argument to minimize must be a tuple, got {}"
        raise TypeError(msg.format(args))

    # Handle args for objective function
    if args:

        def fun_with_args(x):
            return fun(x, *args)

    else:
        fun_with_args = fun

    # Handle args for Jacobian function
    if jac is not None and args:

        def jac_with_args(x):
            return jac(x, *args)

    else:
        jac_with_args = jac

    # Set tolerance if provided
    if tol is not None:
        options = dict(options)
        if "gtol" not in options:
            options["gtol"] = tol

    if method.upper() == "BFGS":
        results = minimize_bfgs(fun_with_args, x0, jac=jac_with_args, bounds=bounds, **options)
        success = results.converged & jnp.logical_not(results.failed)
        return OptimizeResults(
            x=results.x_k,
            success=success,
            status=results.status,
            fun=results.f_k,
            jac=results.g_k,
            hess_inv=results.H_k,
            nfev=results.nfev,
            njev=results.ngev,
            nit=results.k,
        )

    raise ValueError(f"Method {method} not recognized")
