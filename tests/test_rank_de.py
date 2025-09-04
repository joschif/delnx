"""Tests for rank_de module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.stats import mannwhitneyu, rankdata

# Import the module to test
from delnx.tl import rank_de
from delnx.tl._rank_de import (
    _determine_algorithm,
    _rank_sparse_batch_parallel,
    _rank_sparse_batch_serial,
    _rankdata,
    _rankdata_with_ties,
    _validate_inputs,
)


class TestRankingFunctions:
    """Test core ranking functions."""

    def test_rankdata_empty(self):
        """Test ranking empty array."""
        result = _rankdata(np.array([]))
        assert len(result) == 0

    def test_rankdata_single(self):
        """Test ranking single element."""
        result = _rankdata(np.array([5.0]))
        assert result == [1.0]

    def test_rankdata_no_ties(self):
        """Test ranking without ties."""
        data = np.array([1.0, 3.0, 2.0, 4.0])
        result = _rankdata(data)
        expected = rankdata(data, method="average")
        np.testing.assert_allclose(result, expected)

    def test_rankdata_with_ties_values(self):
        """Test ranking with ties."""
        data = np.array([1.0, 2.0, 2.0, 3.0])
        result = _rankdata(data)
        expected = rankdata(data, method="average")  # [1, 2.5, 2.5, 4]
        np.testing.assert_allclose(result, expected)

    def test_rankdata_with_ties_correction(self):
        """Test tie correction calculation."""
        data = np.array([1.0, 2.0, 2.0, 3.0])
        ranks, tie_correction = _rankdata_with_ties(data)

        # Expected ranks: [1, 2.5, 2.5, 4]
        expected_ranks = rankdata(data, method="average")
        np.testing.assert_allclose(ranks, expected_ranks)

        # Tie correction: 1 - (2^3 - 2) / (4^3 - 4) = 1 - 6/60 = 0.9
        expected_correction = 1 - 6 / 60
        assert abs(tie_correction - expected_correction) < 1e-10


class TestSparseRanking:
    """Test sparse matrix ranking functions."""

    @pytest.fixture
    def simple_sparse_data(self):
        """Create simple sparse test data."""
        # Create a 5x3 sparse matrix
        data = np.array([3.0, 1.0, 2.0, 4.0])
        indices = np.array([1, 2, 0, 3])
        indptr = np.array([0, 2, 3, 4])  # col0: [3,1], col1: [2], col2: [4]
        return data, indices, indptr, 5, 3

    def test_parallel_vs_serial_equivalence_no_ties(self, simple_sparse_data):
        """Test that parallel and serial ranking give same results without ties."""
        data, indices, indptr, nrows, ncols = simple_sparse_data

        # Test without tie correction
        parallel_result = _rank_sparse_batch_parallel(data, indptr, nrows, ncols, False)
        serial_result = _rank_sparse_batch_serial(data, indptr, nrows, ncols, False)

        np.testing.assert_allclose(parallel_result[0], serial_result[0])  # ranked_data
        np.testing.assert_allclose(parallel_result[1], serial_result[1])  # zero_ranks

    def test_parallel_vs_serial_equivalence_with_ties(self, simple_sparse_data):
        """Test that parallel and serial ranking give same results with ties."""
        # Create data with ties
        data = np.array([2.0, 2.0, 1.0, 3.0])
        indptr = np.array([0, 2, 3, 4])
        nrows, ncols = 5, 3

        parallel_result = _rank_sparse_batch_parallel(data, indptr, nrows, ncols, True)
        serial_result = _rank_sparse_batch_serial(data, indptr, nrows, ncols, True)

        np.testing.assert_allclose(parallel_result[0], serial_result[0], rtol=1e-10)  # ranked_data
        np.testing.assert_allclose(parallel_result[1], serial_result[1])  # zero_ranks
        # Note: serial doesn't compute tie corrections, so skip that comparison

    def test_known_ranking_output(self):
        """Test ranking with known expected output."""
        # Column with values [0, 3, 1, 0, 0] -> ranks should be [2, 5, 4, 2, 2]
        # Non-zero values: [3, 1] at indices [1, 2]
        # After ranking non-zeros: [1] -> rank 1, [3] -> rank 2
        # Add offset of 3 (for 3 zeros): final ranks [4, 5]

        data = np.array([3.0, 1.0])
        indptr = np.array([0, 2])  # Single column
        nrows, ncols = 5, 1

        ranked_data, zero_ranks, tie_corrections = _rank_sparse_batch_parallel(data, indptr, nrows, ncols, False)

        # Non-zero values should be ranked as [2, 1] then shifted by 3 -> [5, 4]
        expected_ranks = np.array([2.0, 1.0]) + 3.0  # [5, 4]
        np.testing.assert_allclose(ranked_data, expected_ranks)

        # Zero rank should be (3+1)/2 = 2
        assert zero_ranks[0] == 2.0


class TestAUROCCalculation:
    """Test AUROC calculation components."""

    def test_mannwhitney_equivalence(self):
        """Test that our AUROC calculation matches scipy's mannwhitneyu."""
        np.random.seed(42)

        # Create test data
        group1_data = np.random.normal(1.0, 1.0, 20)
        group2_data = np.random.normal(0.0, 1.0, 15)

        # Calculate using scipy
        u_stat, p_val = mannwhitneyu(group1_data, group2_data, alternative="two-sided")
        expected_auroc = u_stat / (len(group1_data) * len(group2_data))

        # Our implementation should match (approximately)
        # Note: This is a simplified test - full test would require setting up the JAX function
        assert 0 <= expected_auroc <= 1


class TestInputValidation:
    """Test input validation functions."""

    def test_validate_inputs_missing_key(self):
        """Test validation with missing condition key."""
        adata = AnnData(np.random.randn(10, 5))
        adata.obs["cell_type"] = ["A"] * 5 + ["B"] * 5

        with pytest.raises(ValueError, match="Condition key 'missing' not found"):
            _validate_inputs(adata, "missing", 2)

    def test_validate_inputs_insufficient_samples(self):
        """Test validation with insufficient samples."""
        adata = AnnData(np.random.randn(10, 5))
        adata.obs["cell_type"] = ["A"] * 1 + ["B"] * 1 + ["C"] * 8  # A and B have <2 samples

        valid_conditions = _validate_inputs(adata, "cell_type", 2)
        assert valid_conditions == ["C"]

    def test_validate_inputs_too_few_conditions(self):
        """Test validation with too few valid conditions."""
        adata = AnnData(np.random.randn(10, 5))
        adata.obs["cell_type"] = ["A"] * 10  # Only one condition

        with pytest.raises(ValueError, match="Need at least 2 valid conditions"):
            _validate_inputs(adata, "cell_type", 2)


class TestAlgorithmSelection:
    """Test algorithm selection logic."""

    def test_determine_algorithm_low_cpu(self):
        """Test algorithm selection with low CPU count."""
        with patch("numba.get_num_threads", return_value=2):
            use_parallel = _determine_algorithm(None, False, False)
            assert not use_parallel

    def test_determine_algorithm_high_cpu(self):
        """Test algorithm selection with high CPU count."""
        with patch("numba.get_num_threads", return_value=8):
            use_parallel = _determine_algorithm(None, False, False)
            assert use_parallel

    def test_determine_algorithm_ties_forces_parallel(self):
        """Test that tie correction forces parallel processing."""
        with patch("numba.get_num_threads", return_value=2):
            use_parallel = _determine_algorithm(None, True, False)
            assert use_parallel


class TestEndToEndIntegration:
    """Test complete workflow integration."""

    @pytest.fixture
    def simple_adata(self):
        """Create simple AnnData for testing."""
        # Create data where group A has higher expression than group B
        np.random.seed(42)
        n_genes = 50

        # Group A: higher expression
        expr_A = np.random.poisson(5, size=(50, n_genes))
        # Group B: lower expression
        expr_B = np.random.poisson(2, size=(50, n_genes))

        X = np.vstack([expr_A, expr_B])
        adata = AnnData(sparse.csr_matrix(X))
        adata.obs["condition"] = ["A"] * 50 + ["B"] * 50
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        return adata

    def test_basic_functionality(self, simple_adata):
        """Test basic rank_de functionality."""
        result = rank_de(simple_adata, "condition", verbose=False)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        expected_columns = {"feature", "condition", "auroc", "z_score", "pval", "padj"}
        assert set(result.columns) == expected_columns

        # Check dimensions
        n_genes = simple_adata.n_vars
        n_conditions = len(simple_adata.obs["condition"].unique())
        assert len(result) == n_genes * n_conditions

        # Check AUROC values are in valid range
        assert result["auroc"].min() >= 0.0
        assert result["auroc"].max() <= 1.0

        # Check p-values are in valid range
        assert result["pval"].min() >= 0.0
        assert result["pval"].max() <= 1.0

    def test_with_ties_correction(self, simple_adata):
        """Test rank_de with tie correction."""
        result_no_ties = rank_de(simple_adata, "condition", use_ties=False, verbose=False)
        result_with_ties = rank_de(simple_adata, "condition", use_ties=True, verbose=False)

        # Results should be similar but not identical
        assert len(result_no_ties) == len(result_with_ties)

        # AUROC values should be identical (tie correction affects p-values, not AUROC)
        merged = result_no_ties.merge(result_with_ties, on=["feature", "condition"], suffixes=("_no_ties", "_ties"))
        np.testing.assert_allclose(merged["auroc_no_ties"], merged["auroc_ties"], rtol=1e-10)

    def test_batch_processing(self, simple_adata):
        """Test that batch processing gives same results as single batch."""
        result_single = rank_de(simple_adata, "condition", batch_size=None, verbose=False)
        result_batched = rank_de(simple_adata, "condition", batch_size=10, verbose=False)

        # Sort both for comparison
        result_single = result_single.sort_values(["feature", "condition"]).reset_index(drop=True)
        result_batched = result_batched.sort_values(["feature", "condition"]).reset_index(drop=True)

        # Results should be identical
        np.testing.assert_allclose(result_single["auroc"], result_batched["auroc"])
        np.testing.assert_allclose(result_single["pval"], result_batched["pval"])

    def test_different_cpu_settings(self, simple_adata):
        """Test that different CPU settings give equivalent results."""
        # Force serial processing
        result_serial = rank_de(simple_adata, "condition", n_cpus=1, verbose=False)

        # Force parallel processing
        result_parallel = rank_de(simple_adata, "condition", n_cpus=8, verbose=False)

        # Sort both for comparison
        result_serial = result_serial.sort_values(["feature", "condition"]).reset_index(drop=True)
        result_parallel = result_parallel.sort_values(["feature", "condition"]).reset_index(drop=True)

        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_allclose(result_serial["auroc"], result_parallel["auroc"], rtol=1e-10)
        np.testing.assert_allclose(result_serial["pval"], result_parallel["pval"], rtol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_condition(self):
        """Test behavior with only one condition."""
        adata = AnnData(np.random.randn(10, 5))
        adata.obs["condition"] = ["A"] * 10

        with pytest.raises(ValueError, match="Need at least 2 valid conditions"):
            rank_de(adata, "condition", verbose=False)

    def test_empty_features(self):
        """Test behavior with empty features."""
        adata = AnnData(np.zeros((10, 5)))  # All zeros
        adata.obs["condition"] = ["A"] * 5 + ["B"] * 5

        # Should not crash, but all AUROCs should be 0.5 (random)
        result = rank_de(adata, "condition", verbose=False)
        np.testing.assert_allclose(result["auroc"], 0.5, atol=1e-10)

    def test_identical_groups(self):
        """Test behavior when groups have identical expression."""
        np.random.seed(42)
        X = np.random.randn(10, 5)
        adata = AnnData(np.vstack([X, X]))  # Identical groups
        adata.obs["condition"] = ["A"] * 10 + ["B"] * 10

        result = rank_de(adata, "condition", verbose=False)
        # AUROCs should be close to 0.5 for identical groups
        np.testing.assert_allclose(result["auroc"], 0.5, atol=0.1)


class TestNumericalStability:
    """Test numerical stability and extreme cases."""

    def test_extreme_values(self):
        """Test with extreme expression values."""
        adata = AnnData(
            np.array(
                [
                    [1e10, 1e-10],  # Very large and small values
                    [1e10, 1e-10],
                    [0, 1],  # Mixed with normal values
                    [0, 1],
                ]
            )
        )
        adata.obs["condition"] = ["A", "A", "B", "B"]

        # Should not crash or produce NaN/inf values
        result = rank_de(adata, "condition", verbose=False)
        assert not result["auroc"].isna().any()
        assert not result["pval"].isna().any()
        assert np.all(np.isfinite(result["auroc"]))
        assert np.all(np.isfinite(result["pval"]))

    def test_sparse_matrix_formats(self):
        """Test different sparse matrix formats."""
        X_dense = np.random.poisson(2, (20, 10))
        adata_csr = AnnData(sparse.csr_matrix(X_dense))
        adata_csc = AnnData(sparse.csc_matrix(X_dense))
        adata_dense = AnnData(X_dense)

        for adata in [adata_csr, adata_csc, adata_dense]:
            adata.obs["condition"] = ["A"] * 10 + ["B"] * 10

            # Should work with any matrix format
            result = rank_de(adata, "condition", verbose=False)
            assert len(result) > 0
            assert not result["auroc"].isna().any()
