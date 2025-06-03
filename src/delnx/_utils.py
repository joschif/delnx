import sys
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from io import StringIO

import numpy as np
from anndata import AnnData
from scipy import sparse


@contextmanager
def suppress_output(verbose: bool = False):
    """Context manager to suppress stdout/stderr and warnings.

    Parameters
    ----------
    verbose
        If True, show all output and warnings. If False, suppress them.
    """
    if verbose:
        yield
    else:
        # Suppress stdout/stderr
        new_stdout, new_stderr = StringIO(), StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = new_stdout, new_stderr

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def aggregate(x: np.ndarray, groups: np.ndarray | None = None, fun: Callable = np.mean, axis: int = 0) -> np.ndarray:
    """Aggregate a matrix by groups.

    Parameters
    ----------
    x : numpy.ndarray
        Input matrix to aggregate.
    groups : numpy.ndarray, optional
        Group labels for aggregation, by default None
    fun : callable, optional
        Aggregation function, by default np.mean
    axis : int, optional
        Axis along which to aggregate, by default 0

    Returns
    -------
    numpy.ndarray
        Aggregated matrix. If groups=None, returns array of aggregated values.
        Otherwise, returns stacked array of aggregated values per group.
    """
    if groups is None:
        return fun(x, axis=axis)
    else:
        return np.stack([fun(x[groups == g], axis=axis) for g in np.unique(groups)])


def _get_layer(adata: AnnData, layer: str | None) -> np.ndarray | sparse.spmatrix:
    """Get data from AnnData layer or X if layer is None.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Layer to use. If None, use adata.X

    Returns
    -------
    array-like
        Data matrix from specified layer or X.
    """
    return adata.layers[layer] if layer is not None else adata.X


def _to_dense(x: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Convert input to dense array.

    Parameters
    ----------
    x : array-like
        Input array or sparse matrix

    Returns
    -------
    numpy.ndarray
        Dense array
    """
    return x.toarray() if sparse.issparse(x) else x


def _to_list(x: Sequence) -> list:
    """Convert input to list.

    Parameters
    ----------
    x : array-like
        Input array or pandas Series

    Returns
    -------
    list
        List of values
    """
    try:
        return x.tolist()
    except AttributeError:
        return list(x)
