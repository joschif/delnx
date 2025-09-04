import anndata as ad
import numpy as np
import pandas as pd
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
    adata_pb.var["dispersion"] = np.random.uniform(0.01, 2.0, size=adata_pb.n_vars)

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


@pytest.fixture
def gene_sets():
    """Load gene sets for testing."""
    return dx.ds.get_gene_sets(collection="hallmark")


@pytest.fixture
def de_results():
    """Create an exemplary de_results DataFrame for testing."""
    # Create sample data with multiple groups and directions using real gene symbols
    data = {
        "group": ["group1"] * 20 + ["group2"] * 20 + ["group3"] * 15,
        "direction": ["up"] * 10 + ["down"] * 10 + ["up"] * 8 + ["down"] * 12 + ["up"] * 7 + ["down"] * 8,
        "auroc": [1] * 55,
        "feature": [
            # Group1 up-regulated genes (immune/inflammation related)
            "TNF",
            "IL1B",
            "IL6",
            "CXCL8",
            "CCL2",
            "NFKB1",
            "JUN",
            "FOS",
            "MYC",
            "STAT3",
            # Group1 down-regulated genes (metabolic)
            "PPARA",
            "PPARG",
            "SREBF1",
            "FASN",
            "ACACA",
            "SCD",
            "ELOVL6",
            "DGAT1",
            "PLIN2",
            "ADIPOQ",
            # Group2 up-regulated genes (cell cycle/proliferation)
            "CCND1",
            "CCNE1",
            "CDK2",
            "CDK4",
            "E2F1",
            "PCNA",
            "MKI67",
            "TOP2A",
            # Group2 down-regulated genes (apoptosis/tumor suppressor)
            "TP53",
            "BAX",
            "CASP3",
            "CASP9",
            "PUMA",
            "BAK1",
            "BID",
            "APAF1",
            "CYCS",
            "BCL2L11",
            "FOXO3",
            "CDKN1A",
            # Group3 up-regulated genes (neuronal)
            "BDNF",
            "NTRK2",
            "CREB1",
            "ARC",
            "EGR1",
            "CAMK2A",
            "GRIN1",
            # Group3 down-regulated genes (stress response)
            "HSP90AA1",
            "HSPA1A",
            "DNAJB1",
            "ATF4",
            "DDIT3",
            "XBP1",
            "ERN1",
            "EIF2AK3",
        ],
        # Optional: Add some additional columns that might be present in real DE results
        "coef": [
            2.1,
            1.8,
            3.2,
            1.5,
            2.9,
            1.7,
            2.3,
            1.9,
            2.6,
            1.4,
            -2.2,
            -1.6,
            -3.1,
            -1.8,
            -2.7,
            -1.9,
            -2.4,
            -1.7,
            -2.8,
            -1.5,
            1.9,
            2.4,
            1.6,
            2.8,
            1.3,
            2.1,
            1.8,
            2.5,
            -1.7,
            -2.3,
            -1.9,
            -2.6,
            -1.4,
            -2.0,
            -1.8,
            -2.4,
            -1.6,
            -2.7,
            -1.5,
            -2.1,
            2.2,
            1.7,
            2.9,
            1.4,
            2.6,
            1.8,
            2.3,
            -2.5,
            -1.9,
            -2.8,
            -1.6,
            -2.2,
            -1.7,
            -2.4,
            -1.8,
        ],
        "pval": [
            0.001,
            0.003,
            0.0001,
            0.005,
            0.0002,
            0.004,
            0.002,
            0.006,
            0.0003,
            0.007,
            0.002,
            0.004,
            0.0001,
            0.003,
            0.0002,
            0.005,
            0.001,
            0.006,
            0.0003,
            0.008,
            0.003,
            0.001,
            0.005,
            0.0002,
            0.009,
            0.002,
            0.004,
            0.0001,
            0.004,
            0.002,
            0.006,
            0.0003,
            0.008,
            0.003,
            0.005,
            0.001,
            0.007,
            0.0002,
            0.009,
            0.002,
            0.001,
            0.005,
            0.0002,
            0.008,
            0.0003,
            0.004,
            0.002,
            0.002,
            0.006,
            0.0001,
            0.007,
            0.003,
            0.005,
            0.001,
            0.004,
        ],
    }

    data_df = pd.DataFrame(data)
    data_df["padj"] = data_df["pval"]

    return data_df
