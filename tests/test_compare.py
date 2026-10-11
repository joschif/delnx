import numpy as np
import pytest

from delnx.tl import compare, de


@pytest.mark.parametrize("mode,reference", [("pairwise", None), ("all_vs_ref", "control")])
def test_compare_with_covariates_tests_correct_contrast(adata_pb_lognorm, mode, reference):
    """Regression test: compare() must test the condition contrast, not a covariate.

    Previously `_run_pairwise` called `func` without a `contrast`, which
    resolved to the last design coefficient -- a covariate whenever
    `covariate_keys` was passed. Verify pairwise/all_vs_ref results match a
    direct `de()` call with the explicit condition contrast.
    """
    adata_pb_lognorm.obs["covariate"] = np.random.rand(adata_pb_lognorm.n_obs)

    results = compare(
        de,
        adata_pb_lognorm,
        condition_key="condition_str",
        mode=mode,
        reference=reference,
        covariate_keys=["covariate"],
    )

    for test_level, ref_level in results[["test", "reference"]].drop_duplicates().itertuples(index=False):
        mask = adata_pb_lognorm.obs["condition_str"].isin([test_level, ref_level]).values
        direct = de(
            adata_pb_lognorm[mask, :].copy(),
            condition_key="condition_str",
            reference=ref_level,
            contrast=f"condition_str[T.{test_level}]",
            covariate_keys=["covariate"],
        )

        subset = results[(results["test"] == test_level) & (results["reference"] == ref_level)]
        actual = subset.sort_values("feature")["pval"].to_numpy()
        expected = direct.sort_values("feature")["pval"].to_numpy()
        np.testing.assert_allclose(actual, expected)
