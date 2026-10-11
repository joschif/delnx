import numpy as np
import pandas as pd
import pytest

from delnx.tl import de


@pytest.mark.parametrize("method", ["lr", "anova", "anova_residual"])
def test_de_methods_pb_lognorm(adata_pb_lognorm, method):
    """Test different DE methods with lognorm pseudobulk data."""
    results = de(
        adata_pb_lognorm,
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
        method=method,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 50
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_default_contrast(adata_pb_lognorm):
    """Test that contrast defaults to last coefficient."""
    results = de(
        adata_pb_lognorm,
        condition_key="condition",
        reference="control",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 50
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_with_covariates(adata_pb_lognorm):
    """Test DE with condition_key + covariate_keys."""
    adata_pb_lognorm.obs["covariate"] = np.random.rand(adata_pb_lognorm.n_obs)

    results = de(
        adata_pb_lognorm,
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
        covariate_keys=["covariate"],
    )
    results_no_cov = de(
        adata_pb_lognorm,
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 50
    # Results should differ when covariates are included
    assert not np.allclose(
        results.set_index("feature")["coef"].values,
        results_no_cov.set_index("feature")["coef"].values,
    )


def test_de_formula_basic(adata_pb_lognorm):
    """Test formula-based de()."""
    results = de(
        adata_pb_lognorm,
        formula="~ condition_str",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_formula_with_covariates(adata_pb_lognorm):
    """Test formula with covariates included directly."""
    results = de(
        adata_pb_lognorm,
        formula="~ condition_str + cell_type",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_formula_continuous(adata_pb_lognorm):
    """Test formula with continuous variable."""
    adata_pb_lognorm.obs["score"] = np.random.normal(size=adata_pb_lognorm.n_obs)

    results = de(
        adata_pb_lognorm,
        formula="~ score",
        contrast="score",
        method="anova",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def _known_effect_adata(d=1.2, n=400, n_genes=20, seed=0):
    """Log1p-like data where the first half of genes is shifted by +d (rest by -d) in 'treat'."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    cond = np.repeat(["control", "treat"], n // 2)
    shift = np.where(np.arange(n_genes) < n_genes // 2, d, -d)
    X = 2.0 + rng.normal(scale=0.8, size=(n, n_genes)) + (cond == "treat")[:, None] * shift
    obs = pd.DataFrame({"condition": cond, "score": rng.normal(size=n)}, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])), shift


def test_de_method_routing_and_log2fc():
    """lr and anova run different tests; log2fc recovers the true log1p mean difference."""
    adata, shift = _known_effect_adata()
    res = {
        m: de(adata, condition_key="condition", reference="control", method=m, verbose=False).set_index("feature")
        for m in ["lr", "anova", "anova_residual"]
    }

    assert not np.allclose(res["lr"]["stat"], res["anova"].loc[res["lr"].index, "stat"])
    expected = pd.Series(shift / np.log(2), index=adata.var_names)
    for r in res.values():
        np.testing.assert_allclose(r["log2fc"], expected[r.index], atol=0.4)
    # Effect size is method-independent
    np.testing.assert_allclose(res["lr"]["coef"], res["anova"].loc[res["lr"].index, "coef"])


def test_de_lr_continuous_contrast_raises():
    """lr needs a binary contrast column."""
    adata, _ = _known_effect_adata()
    with pytest.raises(ValueError, match="binary"):
        de(adata, formula="~ score", contrast="score", method="lr", verbose=False)


def test_de_collinear_contrast_raises():
    """A contrast fully confounded with a covariate has no estimable effect."""
    adata, _ = _known_effect_adata()
    adata.obs["batch"] = adata.obs["condition"].map({"control": "b1", "treat": "b2"})
    with pytest.raises(ValueError, match="collinear"):
        de(adata, formula="~ condition + batch", contrast="condition[T.treat]", method="anova", verbose=False)


@pytest.mark.parametrize(
    "method,layer",
    [
        ("lr", None),
        ("anova", None),
        ("anova_residual", None),
        ("binomial", "binary"),
    ],
)
def test_de_methods_sc(adata_small, method, layer):
    """Test different DE methods on single-cell data."""
    results = de(
        adata_small,
        condition_key="condition",
        reference="control",
        method=method,
        layer=layer,
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_binomial_binary(adata_small):
    """Test binomial DE on binary data."""
    results = de(
        adata_small,
        condition_key="condition",
        reference="control",
        method="binomial",
        layer="binary",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 80
    assert list(results.columns) == ["feature", "log2fc", "coef", "stat", "pval", "padj"]


def test_de_errors(adata_pb_lognorm):
    """Test error conditions."""
    # Both formula and condition_key
    with pytest.raises(ValueError, match="either.*formula.*condition_key"):
        de(adata_pb_lognorm, formula="~ condition_str", condition_key="condition_str")

    # Neither formula nor condition_key
    with pytest.raises(ValueError, match="One of.*formula.*condition_key"):
        de(adata_pb_lognorm)

    # Invalid contrast
    with pytest.raises(ValueError, match="not found in design columns"):
        de(adata_pb_lognorm, formula="~ condition_str", contrast="nonexistent")

    # Unsupported method
    with pytest.raises(ValueError, match="Unsupported method"):
        de(adata_pb_lognorm, condition_key="condition_str", reference="control", method="negbinom")

    # Missing condition_key
    with pytest.raises(ValueError, match="not found in obs"):
        de(adata_pb_lognorm, condition_key="nonexistent")


def test_de_data_type_validation(adata_pb_counts):
    """Test data type validation for different DE methods."""
    # Binomial with non-binary data
    with pytest.raises(ValueError, match="require binary data"):
        de(adata_pb_counts, condition_key="condition", reference="control", method="binomial")

    # ANOVA with count data should work (with warning)
    results = de(adata_pb_counts, condition_key="condition", reference="control", method="anova")
    assert len(results) > 0

    # LR with count data should work
    results = de(adata_pb_counts, condition_key="condition", reference="control", method="lr")
    assert len(results) > 0


def test_grouped_de(adata_pb_lognorm):
    """Test grouped DE via public grouped() wrapper."""
    from delnx.tl import grouped

    results = grouped(
        de,
        adata_pb_lognorm,
        group_key="cell_type",
        condition_key="condition_str",
        reference="control",
        contrast="condition_str[T.treat_a]",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert all(col in results.columns for col in ["feature", "pval", "padj", "group"])
    assert len(results["group"].unique()) == 3
    assert "cell_type_1" in results["group"].values.tolist()


def test_grouped_rank_de(adata_pb_lognorm):
    """Test grouped wrapper with rank_de."""
    from delnx.tl import grouped, rank_de

    results = grouped(
        rank_de,
        adata_pb_lognorm,
        group_key="cell_type",
        condition_key="condition_str",
    )

    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert all(col in results.columns for col in ["feature", "pval", "padj", "group"])
    assert len(results["group"].unique()) == 3
