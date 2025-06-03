from delnx.pp._pseudobulk import pseudobulk


def test_pseudobulk(adata):
    """Test pseudobulk aggregation."""
    import numpy as np

    # Test basic pseudobulk aggregation
    adata_pb = pseudobulk(adata, sample_key="condition_sample", group_key="cell_type", layer="counts")
    assert adata_pb.n_obs < adata.n_obs  # Should have fewer observations
    assert adata_pb.n_vars == adata.n_vars  # Should have same number of variables
    assert "condition_sample" in adata_pb.obs.columns
    assert "cell_type" in adata_pb.obs.columns
    assert "condition" in adata_pb.obs.columns
    assert "sample" in adata_pb.obs.columns

    # Test with different aggregation mode
    adata_pb_mean = pseudobulk(adata, sample_key="condition_sample", group_key="cell_type", mode="mean")
    assert not (adata_pb.X == adata_pb_mean.X).all()  # Should be different from sum

    # Test with count layer
    adata_pb_counts = pseudobulk(adata, sample_key="condition_sample", group_key="cell_type", layer="counts")
    X_flat = adata_pb_counts.X.flatten()
    assert adata_pb_counts.n_obs == adata_pb.n_obs
    assert X_flat.max() > 1000
    assert np.all(np.equal(np.mod(X_flat, 1), 0))
    assert np.all(X_flat >= 0)

    # Test with binary layer
    adata_pb_counts = pseudobulk(
        adata, sample_key="condition_sample", group_key="cell_type", layer="binary", mode="mean"
    )
    X_flat = adata_pb_counts.X.flatten()
    assert adata_pb_counts.n_obs == adata_pb.n_obs
    assert X_flat.max() <= 1
    assert X_flat.min() >= 0
