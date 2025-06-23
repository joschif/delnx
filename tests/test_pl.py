import matplotlib.pyplot as plt

import delnx as dx


def test_plot_volcanoplot(adata_pb_counts):
    """Test plotting of matrixplot."""

    # Compute size factors
    dx.pp.size_factors(adata_pb_counts, method="ratio")

    # Estimate dispersion parameters
    dx.pp.dispersion(adata_pb_counts, size_factor_key="size_factor", method="deseq2")

    # Run differential expression analysis
    results = dx.tl.de(
        adata_pb_counts,
        condition_key="condition",
        group_key="cell_type",
        mode="all_vs_ref",
        reference="False",
        method="negbinom",
        layer="counts",
        size_factor_key="size_factor",
        dispersion_key="dispersion",
    )

    dx.pp.label_de_genes(results, coef_thresh=0.5)

    fig = dx.pl.volcanoplot(results, label_top=5, coef_thresh=0.5)

    assert fig is not None
    assert isinstance(fig.fig, plt.Figure)
