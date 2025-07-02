import marsilea as ma
import matplotlib.pyplot as plt
import numpy as np

import delnx as dx


def test_plot_volcanoplot(adata_pb_counts):
    """Test plotting of volcanoplot."""

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
        reference="control",
        method="negbinom",
        size_factor_key="size_factor",
        dispersion_key="dispersion",
    )

    # Label differentially expressed genes
    dx.pp.label_de_genes(results, coef_thresh=0.5)

    # Select one cell type at a time
    results = results[results["group"] == "cell_type_1"]

    # Plot volcano plot
    fig, _ = dx.pl.volcanoplot(
        results, thresh={"coef": 2, "-log10(pval)": -np.log10(0.05)}, label_top=5, return_fig=True
    )

    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, plt.Figure)


def test_plot_violinplot(adata_small):
    """Test plotting of violinplot."""

    plot = dx.pl.ViolinPlot(
        adata_small,
        genes=["gene_1", "gene_2", "gene_3"],
        groupby="cell_type",
        splitby="condition",
        include_groups=["cell_type_1"],
        include_splits=["ctr", "treatment"],
        figsize=(2, 4),
        use_raw=False,
        flip=True,
    )
    board = plot._build_plot(plot_type="violin")
    # Check if the figure is created
    assert board is not None
    assert isinstance(board, ma.base.StackBoard)


def test_matrixplot(adata_small):
    """Test plotting of matrixplot."""

    m = dx.pl.MatrixPlot(
        adata_small,
        markers=["gene_1", "gene_2", "gene_3"],
        groupby_keys=["cell_type", "condition"],
        group_names=["Cell type", "Condition"],
        row_grouping=["cell_type", "condition"],
        column_grouping=True,
        dendrograms=["left"],
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        center=0,
        width=10,
        height=10,
        scale=1,
        show_legends=True,
        show_column_names=False,
        show_row_names=True,
    )
    fig = m._build_plot()
    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, ma.heatmap.Heatmap)


def test_dotplot(adata_small):
    """Test plotting of matrixplot."""

    m = dx.pl.DotPlot(
        adata_small,
        markers=["gene_1", "gene_2", "gene_3"],
        groupby_keys=["cell_type", "condition"],
        group_names=["Cell type", "Condition"],
        row_grouping=["cell_type", "condition"],
        column_grouping=True,
        dendrograms=["left"],
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        center=0,
        width=10,
        height=10,
        scale=1,
        show_legends=True,
        show_column_names=False,
        show_row_names=True,
    )
    fig = m._build_plot()
    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, ma.heatmap.SizedHeatmap)


def test_gsea_barplot(de_results, gene_sets):
    """Test that gsea_barplot returns a valid figure."""

    # Label DE genes
    dx.pp.label_de_genes(de_results, coef_thresh=0.5)

    # Run enrichment analysis
    enr_results = dx.tl.de_enrichment_analysis(de_results, gene_sets=gene_sets, cutoff=0.1)

    # Plot GSEA barplot
    fig = dx.pl.gsea_barplot(enr_results, group_key=["group", "up_dw"], top_n=10)

    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, ma.base.ClusterBoard)


def test_gsea_dotplot(de_results, gene_sets):
    """Test that gsea_dotplot returns a valid figure."""

    # Label DE genes
    dx.pp.label_de_genes(de_results, coef_thresh=0.5)

    # Run enrichment analysis
    enr_results = dx.tl.de_enrichment_analysis(de_results, gene_sets=gene_sets, cutoff=0.1)

    # Plot GSEA dotplot
    fig = dx.pl.gsea_dotplot(enr_results, group_key=["group", "up_dw"], top_n=10)

    # Check if the figure is created
    assert fig is not None
    assert isinstance(fig, ma.heatmap.SizedHeatmap)
