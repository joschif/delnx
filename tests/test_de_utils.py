"""Tests for utility functions in the DE module."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_data():
    # Create example data for each type
    rng = np.random.RandomState(0)
    counts = rng.negative_binomial(n=20, p=0.3, size=(10, 5))
    lognorm = np.log1p(counts / counts.sum(axis=0, keepdims=True))
    binary = (counts > counts.mean()).astype(float)
    return {"counts": counts, "lognorm": lognorm, "binary": binary}


def test_infer_data_type(test_data):
    """Test data type inference."""
    from orchard.tl._utils import _infer_data_type

    # Test raw counts detection
    assert _infer_data_type(test_data["counts"]) == "counts"

    # Test log-normalized detection
    assert _infer_data_type(test_data["lognorm"]) == "lognorm"

    # Test binary detection
    assert _infer_data_type(test_data["binary"]) == "binary"


@pytest.mark.parametrize("data_type", ["counts", "lognorm", "binary"])
def test_log2fc(test_data, data_type):
    """Test log2fc calculation for different data types."""
    from orchard.tl._effects import _log2fc

    # Create reference mask (first half ref, second half test)
    ref_mask = np.array([True] * 5 + [False] * 5)

    # Calculate log2fc for each data type
    for _, X in test_data.items():
        log2fc = _log2fc(X, ref_mask, data_type=data_type)

        # Basic checks
        assert isinstance(log2fc, np.ndarray)
        assert log2fc.shape == (X.shape[1],)
        assert not np.any(np.isnan(log2fc))
        assert not np.any(np.isinf(log2fc))

        # Check values match expectations
        eps = 1e-8
        if data_type == "lognorm":
            # For log1p data, first reverse transform then calculate ratios
            ref_counts = np.expm1(X[~ref_mask])
            test_counts = np.expm1(X[ref_mask])
            expected = np.log2((test_counts.mean(axis=0) + eps) / (ref_counts.mean(axis=0) + eps))
        else:
            # For counts and binary, calculate ratio of means directly
            expected = np.log2((X[ref_mask].mean(axis=0) + eps) / (X[~ref_mask].mean(axis=0) + eps))
        np.testing.assert_allclose(log2fc, expected)


def test_auroc(test_data):
    """Test AUROC calculation."""
    from orchard.tl._effects import _batched_auroc

    data = test_data["counts"]
    labels = np.array([1] * 5 + [0] * 5)  # First half positive, second half negative
    auroc = _batched_auroc(data, labels, batch_size=2)

    assert isinstance(auroc, np.ndarray)
    assert len(auroc) == data.shape[1]
    assert auroc.min() >= 0.0
    assert auroc.max() <= 1.0


def test_invalid_data_type():
    """Test error handling for invalid data type."""
    from orchard.tl._effects import _log2fc

    X = np.random.randn(10, 5)
    ref_mask = np.array([True] * 5 + [False] * 5)

    with pytest.raises(ValueError, match="Unsupported data type"):
        _log2fc(X, ref_mask, data_type="invalid")


@pytest.mark.parametrize(
    "conditions",
    [
        np.array(["A", "A", "B", "B", "C", "C"]),
        pd.Series(["A", "A", "B", "B", "C", "C"]),
        pd.Categorical(["A", "A", "B", "B", "C", "C"]),
    ],
)
def test_validate_conditions(conditions):
    """Test condition validation for different modes."""
    from orchard.tl._utils import _validate_conditions

    # Test all_vs_ref mode
    levels, comps = _validate_conditions(conditions, reference="A", mode="all_vs_ref")
    assert sorted(levels) == ["A", "B", "C"]
    assert sorted(comps) == sorted([("B", "A"), ("C", "A")])

    # Test all_vs_all mode
    levels, comps = _validate_conditions(conditions, mode="all_vs_all")
    assert sorted(levels) == ["A", "B", "C"]
    assert sorted(comps) == sorted([("A", "B"), ("A", "C"), ("B", "C")])

    # Test pairwise mode
    levels, comps = _validate_conditions(conditions, reference=("A", "B"), mode="1_vs_1")
    assert sorted(levels) == ["A", "B", "C"]
    assert comps == [("B", "A")]

    # Test binary conditions
    binary = (conditions == "A").astype(bool)
    levels, comps = _validate_conditions(binary, reference=True, mode="all_vs_ref")
    assert sorted(levels) == [False, True]
    assert comps == [(False, True)]

    # Test error cases
    with pytest.raises(ValueError, match="Need at least 2 condition levels"):
        _validate_conditions(np.array(["A", "A", "A"]))

    with pytest.raises(ValueError, match="Reference.*not in levels"):
        _validate_conditions(conditions, reference="D", mode="all_vs_ref")

    with pytest.raises(ValueError, match="must be a tuple"):
        _validate_conditions(conditions, reference=None, mode="1_vs_1")
