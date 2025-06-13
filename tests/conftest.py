import anndata as ad
import numpy as np
import pytest

import delnx as dx


@pytest.fixture
def adata():
    """Create test data for general testing."""
    # Use our synthetic data generator with adjusted parameters
    n_cells = 3000
    n_samples = 3
    n_genes = 100

    adata1 = dx.ds.synthetic_adata(
        n_cells=n_cells,
        n_genes=n_genes,
        n_cell_types=3,
        n_de_genes=10,
        mean_counts=20,
        dispersion=0.3,
        dropout_rate=0.3,
        n_samples=n_samples,
        random_seed=42,
    )

    adata2 = dx.ds.synthetic_adata(
        n_cells=n_cells,
        n_genes=n_genes,
        n_cell_types=3,
        n_de_genes=10,
        mean_counts=20,
        dispersion=0.3,
        dropout_rate=0.3,
        n_samples=n_samples,
        random_seed=43,
    )

    adata1.obs["condition_str"] = np.where(adata1.obs["condition"] != "control", "treat_a", adata1.obs["condition"])
    adata2.obs["condition_str"] = np.where(adata2.obs["condition"] != "control", "treat_b", adata2.obs["condition"])

    # Concatenate the two datasets
    adata = ad.concat([adata1, adata2], axis=0)
    adata.obs.index = "cell_" + np.arange(adata.n_obs).astype(str)

    # Add some additional metadata for testing
    adata.obs["continuous_covar"] = np.random.normal(size=adata.n_obs)
    adata.obs["condition_sample"] = adata.obs["condition_str"] + "_" + adata.obs["sample"].astype(str)
    adata.obs["condition_bool"] = np.where(adata.obs["condition"] == "control", False, True)
    adata.obs["condition_int"] = adata.obs["condition_bool"].astype(int)
    adata.obs["condition_float"] = adata.obs["condition_int"].astype(float)
    adata.obs["condition_cat"] = adata.obs["condition_str"].astype("category")

    adata.layers["binary"] = adata.X.copy()
    adata.layers["binary"] = (adata.X > 0).astype(int)

    return adata


@pytest.fixture
def adata_small():
    """Create small test data for testing on the single-cell level."""
    # Use our synthetic data generator with adjusted parameters
    n_cells = 500
    n_genes = 100
    n_samples = 2

    adata = dx.ds.synthetic_adata(
        n_cells=n_cells,
        n_genes=n_genes,
        n_cell_types=3,
        n_de_genes=10,
        mean_counts=20,
        dispersion=0.3,
        dropout_rate=0.3,
        n_samples=n_samples,
        random_seed=42,
    )

    # Add some additional metadata for testing
    adata.obs["continuous_covar"] = np.random.normal(size=adata.n_obs)
    adata.layers["binary"] = adata.X.copy()
    adata.layers["binary"] = (adata.layers["binary"] > 0).astype(int)

    return adata


@pytest.fixture
def adata_pb_counts(adata):
    """Create pseudobulk data for testing."""
    # Create pseudobulk data
    adata_pb = dx.pp.pseudobulk(
        adata,
        sample_key="condition_sample",
        group_key="cell_type",
        layer="counts",
        mode="sum",
    )
    adata_pb.obs["size_factors"] = adata_pb.obs["psbulk_counts"] / adata_pb.obs["psbulk_counts"].mean()

    return adata_pb


@pytest.fixture
def adata_pb_lognorm(adata):
    """Create pseudobulk data for testing."""
    # Create pseudobulk data
    adata_pb = dx.pp.pseudobulk(
        adata,
        sample_key="condition_sample",
        group_key="cell_type",
        mode="mean",
    )

    return adata_pb
