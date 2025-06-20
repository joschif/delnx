from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from delnx._typing import ComparisonMode, DataType
from delnx._utils import _to_dense, _to_list

import warnings
from typing import TYPE_CHECKING

from .. import _logger as logg
from .._settings import settings
from .._utils import _choose_graph

if TYPE_CHECKING:
    from anndata import AnnData

    from .._compat import CSRBase, SpBase


def _infer_data_type(X: np.ndarray) -> DataType:
    """Infer the type of data from its values.

    Parameters
    ----------
    X
        Expression matrix

    Returns
    -------
    DataType
        Inferred data type:
        - counts: Raw count data (integers, potentially large values)
        - lognorm: Log-normalized data (floating point, typically between 0 and 10)
        - binary: Binary data (only 0s and 1s)
    """
    # Subsample cells if large
    if X.shape[0] > 300:
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], size=300, replace=False)
        sample = X[idx, :]
    else:
        sample = X

    sample = _to_dense(sample).flatten()

    # Check if binary
    unique_vals = np.unique(sample)
    if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
        return "binary"

    # Check if counts (all non-negative integers)
    is_integer = np.all(np.equal(np.mod(sample, 1), 0))
    is_nonnegative = np.all(sample >= 0)
    if is_integer and is_nonnegative:
        return "counts"

    # Otherwise assume log-normalized
    return "lognorm"


def _validate_conditions(
    condition_values: np.ndarray | pd.Series | pd.Categorical,
    reference: str | tuple[str, str] | None = None,
    mode: ComparisonMode = "all_vs_ref",
) -> tuple[list[str], list[tuple[str, str]]]:
    """Validate condition values and return valid comparisons.

    Parameters
    ----------
    condition_values
        Array or Series of condition values
    reference
        Reference level for comparisons or tuple of (ref_group, comp_group)
    mode
        How to perform comparisons:
        - all_vs_ref: Compare all levels to reference
        - all_vs_all: Compare all pairs of levels
        - 1_vs_1: Compare only two levels (reference and comparison group)

    Returns
    -------
    tuple
        levels: List of unique condition levels
        comparisons: List of tuples (treatment, reference) to compare
    """
    # Get unique levels
    levels = sorted(set(_to_list(condition_values)))

    if len(levels) < 2:
        raise ValueError(f"Need at least 2 condition levels, got {len(levels)}: {levels}")

    # Handle different modes
    # Unpack reference if it's a tuple
    ref = None
    alt = None
    if isinstance(reference, tuple):
        ref, alt = reference
    else:
        ref = reference

    if mode == "1_vs_1":
        if not isinstance(reference, tuple):
            raise ValueError("For 1_vs_1 mode, `reference` must be a tuple (ref_group, comp_group)")
        if ref is None or alt is None:
            raise ValueError("For 1_vs_1 mode, both reference and comparison group must be specified")
        if ref not in levels or alt not in levels:
            raise ValueError(f"Reference '{ref}' and comparison group '{alt}' must be in levels: {levels}")
        comparisons = [(alt, ref)]

    elif mode == "all_vs_ref":
        if ref is None:
            raise ValueError("For all_vs_ref mode, reference must be specified")
        elif ref not in levels:
            raise ValueError(f"Reference '{ref}' not in levels: {levels}")
        comparisons = [(level, ref) for level in levels if level != ref]

    elif mode == "all_vs_all":
        comparisons = [(l1, l2) for i, l1 in enumerate(levels) for l2 in levels[i + 1 :]]

    else:
        raise ValueError(f"Invalid comparison mode: {mode}")

    return list(levels), comparisons


def _prepare_model_data(
    adata: AnnData,
    condition_key: str,
    reference: str,
    covariate_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare data frame for fitting models."""
    model_data = pd.DataFrame(index=range(adata.n_obs))

    # Set up condition
    model_data[condition_key] = (adata.obs[condition_key].values != reference).astype(int)

    # Add covariates
    if covariate_keys is not None:
        for cov in covariate_keys:
            model_data[cov] = adata.obs[cov].values

    return model_data


def _choose_representation(
    adata: AnnData,
    *,
    use_rep: str | None = None,
    n_pcs: int | None = None,
    silent: bool = False,
) -> np.ndarray | CSRBase:  # TODO: what else?
    verbosity = settings.verbosity
    if silent and settings.verbosity > 1:
        settings.verbosity = 1
    if use_rep is None and n_pcs == 0:  # backwards compat for specifying `.X`
        use_rep = "X"
    if use_rep is None:
        X = _get_pca_or_small_x(adata, n_pcs)
    elif use_rep in adata.obsm and n_pcs is not None:
        if n_pcs > adata.obsm[use_rep].shape[1]:
            msg = (
                f"{use_rep} does not have enough Dimensions. Provide a "
                "Representation with equal or more dimensions than"
                "`n_pcs` or lower `n_pcs` "
            )
            raise ValueError(msg)
        X = adata.obsm[use_rep][:, :n_pcs]
    elif use_rep in adata.obsm and n_pcs is None:
        X = adata.obsm[use_rep]
    elif use_rep == "X":
        X = adata.X
    else:
        msg = f"Did not find {use_rep} in `.obsm.keys()`. You need to compute it first."
        raise ValueError(msg)
    settings.verbosity = verbosity  # resetting verbosity
    return X


def _get_pca_or_small_x(adata: AnnData, n_pcs: int | None) -> np.ndarray | CSRBase:
    if adata.n_vars <= settings.N_PCS:
        logg.info("    using data matrix X directly")
        return adata.X

    if "X_pca" in adata.obsm:
        if n_pcs is not None and n_pcs > adata.obsm["X_pca"].shape[1]:
            msg = "`X_pca` does not have enough PCs. Rerun `sc.pp.pca` with adjusted `n_comps`."
            raise ValueError(msg)
        X = adata.obsm["X_pca"][:, :n_pcs]
        logg.info(f"    using 'X_pca' with n_pcs = {X.shape[1]}")
        return X

    from scanpy.pp import pca

    warnings.warn(
        f"You’re trying to run this on {adata.n_vars} dimensions of `.X`, "
        "if you really want this, set `use_rep='X'`.\n         "
        "Falling back to preprocessing with `sc.pp.pca` and default params.",
        stacklevel=3,
    )
    n_pcs_pca = n_pcs if n_pcs is not None else settings.N_PCS
    pca(adata, n_comps=n_pcs_pca)
    return adata.obsm["X_pca"]


def get_init_pos_from_paga(
    adata: AnnData,
    adjacency: SpBase | None = None,
    random_state=0,
    neighbors_key: str | None = None,
    obsp: str | None = None,
):
    np.random.seed(random_state)
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if "pos" not in adata.uns.get("paga", {}):
        msg = "Plot PAGA first, so that `adata.uns['paga']['pos']` exists."
        raise ValueError(msg)

    groups = adata.obs[adata.uns["paga"]["groups"]]
    pos = adata.uns["paga"]["pos"]
    connectivities_coarse = adata.uns["paga"]["connectivities"]
    init_pos = np.ones((adjacency.shape[0], 2))
    for i, group_pos in enumerate(pos):
        subset = (groups == groups.cat.categories[i]).values
        neighbors = connectivities_coarse[i].nonzero()
        if len(neighbors[1]) > 0:
            connectivities = connectivities_coarse[i][neighbors]
            nearest_neighbor = neighbors[1][np.argmax(connectivities)]
            noise = np.random.random((len(subset[subset]), 2))
            dist = group_pos - pos[nearest_neighbor]
            noise = noise * dist
            init_pos[subset] = group_pos - 0.5 * dist + noise
        else:
            init_pos[subset] = group_pos
    return init_pos