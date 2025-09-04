"""Utility functions and classes."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from io import StringIO
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

LegacyUnionType = type(Union[int, str])  # noqa: UP007


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


def get_de_genes(
    df: pd.DataFrame,
    group_key: str = "group",
    effect_key: str = "log2fc",
    pval_key: str = "pval",
    feature_key: str = "feature",
    effect_thresh: float = 1.0,
    pval_thresh: float = 0.01,
    top_n: int | None = None,
    return_labeled_df: bool = False,
) -> dict[str, dict[str, list]] | tuple[dict[str, dict[str, list]], pd.DataFrame]:
    """
    Analyze differential expression data: label significant genes and extract lists per group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing differential expression results.
    group_key : str, default="group"
        Column indicating the group or condition.
    effect_key : str, default="log2fc"
        Column with effect size values (e.g., log2 fold change, coefficient).
    pval_key : str, default="pval"
        Column with p-values.
    feature_key : str, default="feature"
        Column containing gene names.
    effect_thresh : float, default=1.0
        Threshold for absolute effect size.
    pval_thresh : float, default=0.01
        Threshold for significance.
    top_n : int or None, default=None
        Number of top genes to select per direction. If None, return all.
    return_labeled_df : bool, default=False
        If True, also return the labeled DataFrame.

    Returns
    -------
    dict or tuple
        If return_labeled_df is False:
            Dictionary of the form:
            {
                "group1": {"up": [...], "down": [...]},
                "group2": {"up": [...], "down": [...]},
                ...
            }
        If return_labeled_df is True:
            Tuple of (gene_lists_dict, labeled_dataframe)
    """
    # Create a copy to avoid modifying the original dataframe
    df_labeled = df.copy()

    # Add significance labels
    log_pval = -np.log10(df_labeled[pval_key])
    df_labeled["-log10(pval)"] = np.clip(log_pval, a_min=None, a_max=50)

    # Determine significance based on p-value and effect size thresholds
    sig_pval_mask = df_labeled[pval_key] < pval_thresh
    up_mask = sig_pval_mask & (df_labeled[effect_key] > effect_thresh)
    down_mask = sig_pval_mask & (df_labeled[effect_key] < -effect_thresh)

    df_labeled["significant"] = "NS"
    df_labeled.loc[up_mask, "significant"] = "Up"
    df_labeled.loc[down_mask, "significant"] = "Down"

    # Extract DE genes per group
    result = {}
    grouped = df_labeled.groupby(group_key)

    for group, sub_df in grouped:
        up_df = sub_df[sub_df["significant"] == "Up"]
        down_df = sub_df[sub_df["significant"] == "Down"]

        if top_n is not None:
            up_df = up_df.nlargest(top_n, effect_key)
            down_df = down_df.nsmallest(top_n, effect_key)

        result[group] = {
            "up": up_df[feature_key].tolist(),
            "down": down_df[feature_key].tolist(),
        }

    if return_labeled_df:
        return result, df_labeled
    else:
        return result
