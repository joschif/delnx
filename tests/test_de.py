import numpy as np
import pandas as pd
import pytest

from delnx.tl import de, grouped_de


@pytest.mark.parametrize(
    "method,backend",
    [
        # "deseq2",
        ["negbinom", "statsmodels"],
        ["negbinom", "jax"],
    ],
)
def test_de_methods_pb_counts(adata_pb_counts, method):
    """Test different DE methods with appropriate data types."""
    # Run DE analysis
    de_results = de(
        adata_pb_counts,
        condition_key="condition",
        method=method,
        reference="control",
    )

    # Basic checks
    assert isinstance(de_results, pd.DataFrame)
    assert len(de_results) > 0
    assert len(de_results) > 50
    assert all(col in de_results.columns for col in ["feature", "pval", "padj"])


@pytest.mark.parametrize(
    "method,backend",
    [
        ("negbinom", "statsmodels"),
        ("negbinom", "jax"),
        ("lr", "statsmodels"),
        ("lr", "jax"),
        ("anova", "statsmodels"),
        ("anova", "jax"),
        ("anova_residual", "statsmodels"),
        ("anova_residual", "jax"),
    ],
)
def test_de_methods_pb_lognorm(adata_pb_lognorm, method, backend):
    """Test different DE methods with appropriate data types."""
    # Run DE analysis
    de_results = de(
        adata_pb_lognorm,
        condition_key="condition",
        method=method,
        backend=backend,
        reference="control",
    )

    # Basic checks
    assert isinstance(de_results, pd.DataFrame)
    assert len(de_results) > 0
    assert len(de_results) > 50
    assert all(col in de_results.columns for col in ["feature", "pval", "padj"])


@pytest.mark.parametrize(
    "method,backend,layer",
    [
        ("negbinom", "statsmodels", "counts"),
        ("negbinom", "jax", "counts"),
        ("lr", "statsmodels", None),
        ("lr", "jax", None),
        ("anova", "statsmodels", None),
        ("anova", "jax", None),
        ("anova_residual", "statsmodels", None),
        ("anova_residual", "jax", None),
        ("binomial", "statsmodels", "binary"),
    ],
)
def test_de_methods_sc(adata_small, method, backend, layer):
    """Test different DE methods with appropriate data types."""
    # Run DE analysis
    de_results = de(
        adata_small,
        condition_key="condition",
        method=method,
        backend=backend,
        reference="control",
        layer=layer,
    )

    # Basic checks
    assert isinstance(de_results, pd.DataFrame)
    assert len(de_results) > 0
    assert all(col in de_results.columns for col in ["feature", "pval", "padj"])


@pytest.mark.parametrize(
    "condition_key,reference,mode",
    [
        ("condition_str", "control", "all_vs_ref"),  # Test string multi-level vs reference
        ("condition_cat", "control", "all_vs_ref"),  # Test string multi-level vs reference
        ("condition_str", None, "all_vs_all"),  # Test all vs all
        ("condition", None, "all_vs_all"),  # Test all vs all
        (
            "condition_str",
            ("control", "treat_b"),
            "1_vs_1",
        ),  # Tests 1 vs 1 comparison
        (
            "condition_str",
            ("treat_a", "treat_b"),
            "1_vs_1",
        ),  # Tests 1 vs 1 comparison
        ("condition_bool", True, "all_vs_ref"),  # Test boolean conditions
        ("condition_int", 0, "all_vs_ref"),  # Test integer categories
        ("condition_float", 0.0, "all_vs_ref"),  # Test float categories
    ],
)
def test_de_condition_types(adata_pb_lognorm, condition_key, reference, mode):
    """Test DE analysis with different condition types and comparison modes."""
    # Print out the condition values for debugging
    print("\nCondition values:\n", adata_pb_lognorm.obs[condition_key].value_counts())
    # Run DE analysis
    de_results = de(
        adata_pb_lognorm,
        condition_key=condition_key,
        method="lr",  # Use logistic regression as it works with all data types
        reference=reference,
        mode=mode if mode is not None else "all_vs_ref",
    )

    # Basic checks
    assert isinstance(de_results, pd.DataFrame)
    assert len(de_results) > 0
    assert len(de_results) > 50
    assert all(col in de_results.columns for col in ["feature", "pval", "padj", "ref_condition", "test_condition"])

    # Check comparison-specific results
    if mode == "1_vs_1":
        # For tuple reference, should only have one comparison
        assert (de_results["ref_condition"] == reference[0]).all()
        assert (de_results["test_condition"] == reference[1]).all()
    elif mode == "all_vs_all":
        # For all_vs_all, should have all pairwise comparisons
        n_levels = len(adata_pb_lognorm.obs[condition_key].unique())
        expected_comparisons = (n_levels * (n_levels - 1)) // 2
        assert (
            len(pd.unique(de_results[["ref_condition", "test_condition"]].apply(tuple, axis=1))) == expected_comparisons
        )
    elif mode == "all_vs_ref":
        # For vs_ref, should compare all other levels to reference
        assert all(de_results["ref_condition"] == reference)
        other_levels = set(adata_pb_lognorm.obs[condition_key].unique()) - {reference}
        assert set(de_results["test_condition"].unique()) == other_levels


def test_de_errors(adata_pb_counts):
    """Test error conditions in DE analysis."""
    # Test invalid reference level
    with pytest.raises(ValueError, match="Reference.*not in levels"):
        de(adata_pb_counts, condition_key="condition_str", reference="invalid", mode="all_vs_ref")

    with pytest.raises(ValueError, match="must be a tuple.*"):
        de(adata_pb_counts, condition_key="condition_str", reference="invalid", mode="1_vs_1")

    with pytest.raises(ValueError, match="Reference.*must be in levels"):
        de(adata_pb_counts, condition_key="condition_str", reference=("invalid", "treat_b"), mode="1_vs_1")

    # Test not provided reference
    with pytest.raises(ValueError, match="reference.*must be specified"):
        de(adata_pb_counts, condition_key="condition_str", reference=(None, "treat_b"), mode="1_vs_1")

    with pytest.raises(ValueError, match="must be a tuple.*"):
        de(adata_pb_counts, condition_key="condition_str", reference=None, mode="1_vs_1")

    with pytest.raises(ValueError, match="reference.*must be specified"):
        de(adata_pb_counts, condition_key="condition_str", reference=None, mode="all_vs_ref")

    # Test invalid mode
    with pytest.raises(ValueError, match="Invalid comparison mode"):
        de(adata_pb_counts, condition_key="condition_str", reference="control", mode="invalid_mode")

    # Test single level condition
    adata_single = adata_pb_counts.copy()
    adata_single.obs["single_level"] = "same"
    with pytest.raises(ValueError, match="Need at least 2 condition levels"):
        de(adata_single, condition_key="single_level", reference=None)


def test_de_binomial_binary(adata_small):
    """Test binomial DE method with binary data."""
    # Run binomial DE analysis
    de_results = de(
        adata_small,
        condition_key="condition",
        method="binomial",
        reference="control",
        data_type="binary",
        layer="binary",
        log2fc_threshold=0.0,
    )

    # Basic checks
    assert isinstance(de_results, pd.DataFrame)
    assert len(de_results) > 0
    assert len(de_results) > 80
    assert all(col in de_results.columns for col in ["feature", "pval", "padj"])


def test_de_data_type_validation(adata_pb_counts):
    """Test data type validation for different DE methods."""
    # Make a copy of the data
    adata = adata_pb_counts.copy()

    # # Test DESeq2 with non-count data
    # with pytest.raises(ValueError, match="requires count data"):
    #     adata.X = np.log1p(adata.X)  # Convert to log-normalized
    #     de(
    #         adata,
    #         condition_key="condition",
    #         method="deseq2",
    #         reference="control",
    #     )

    # Test binomial with non-binary data
    with pytest.raises(ValueError, match="require binary data"):
        de(
            adata_pb_counts,
            condition_key="condition",
            method="binomial",
            reference="control",
        )

    # Test negative binomial with non-count data
    with pytest.raises(ValueError, match="require count data"):
        adata.X = np.log1p(adata.X)  # Convert to log-normalized
        de(
            adata,
            condition_key="condition",
            method="negbinom",
            reference="control",
        )

    # Test ANOVA with non-log-normalized data
    with pytest.warns(UserWarning, match="ANOVA is designed for log-normalized data"):
        de(
            adata_pb_counts,
            condition_key="condition",
            method="anova",
            reference="control",
        )

    # Test logistic regression with count data
    with pytest.warns(UserWarning, match="designed for log-normalized or binary data"):
        de(
            adata_pb_counts,
            condition_key="condition",
            method="lr",
            reference="control",
        )


@pytest.mark.parametrize(
    "method",
    ["lr"],
)
@pytest.mark.parametrize(
    "mode,reference",
    [
        ("all_vs_ref", "control"),
        ("all_vs_all", None),
        ("1_vs_1", ("control", "treat_a")),
        ("all_vs_ref", "control"),
        ("all_vs_all", None),
        ("1_vs_1", ("control", "treat_a")),
    ],
)
def test_grouped_de(adata_pb_lognorm, method, mode, reference):
    """Test grouped DE analysis."""
    # Run DE analysis with grouping
    de_results = grouped_de(
        adata_pb_lognorm,
        condition_key="condition_str",
        group_key="cell_type",
        method=method,
        backend="statsmodels",
        mode=mode,
        reference=reference,
    )

    # Basic checks
    assert isinstance(de_results, pd.DataFrame)
    assert len(de_results) > 0
    assert len(de_results) > 200
    assert all(col in de_results.columns for col in ["feature", "pval", "padj", "group"])
    assert len(de_results["group"].unique()) == 3
    assert "cell_type_1" in de_results["group"].values.tolist()

    # Run DE analysis with grouping
    de_results = de(
        adata_pb_lognorm,
        condition_key="condition_str",
        group_key="cell_type",
        method=method,
        mode=mode,
        reference=reference,
    )

    # Basic checks
    assert isinstance(de_results, pd.DataFrame)
    assert len(de_results) > 0
    assert len(de_results) > 200
    assert all(col in de_results.columns for col in ["feature", "pval", "padj", "group"])
    assert len(de_results["group"].unique()) == 3
    assert "cell_type_1" in de_results["group"].values.tolist()


@pytest.mark.parametrize(
    "method",
    ["lr", "negbinom"],
)
@pytest.mark.parametrize(
    "optimizer",
    ["BFGS", "IRLS"],
)
def test_jax_optimizers(adata_pb_lognorm, adata_pb_counts, method, optimizer):
    """Test different optimizers for DE analysis."""
    adata = adata_pb_lognorm if method == "lr" else adata_pb_counts

    # Test with BFGS optimizer
    de_results_bfgs = de(
        adata,
        condition_key="condition_str",
        method=method,
        backend="jax",
        optimizer=optimizer,
        reference="control",
    )
    assert isinstance(de_results_bfgs, pd.DataFrame)
    assert len(de_results_bfgs) > 0
    assert all(col in de_results_bfgs.columns for col in ["feature", "pval", "padj"])
    assert de_results_bfgs["pval"].min() >= 0.0
    assert de_results_bfgs["pval"].max() <= 1.0
