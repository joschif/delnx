import matplotlib.pyplot as plt

import delnx as dx
from delnx.pl._utils import process_de, select_top_genes


def test_plot_matrixplot(adata_small):
    """Test plotting of matrixplot."""
    """Test that matrixplot generates a figure without error."""
    de_results = dx.tl.de(
        adata_small,
        condition_key="condition",
        method="binomial",
        backend="statsmodels",
        reference="control",
        data_type="binary",
        layer="binary",
        log2fc_threshold=0.0,
    )
    de_results = process_de(de_results)
    top_up, top_down = select_top_genes(de_results, top_n=50)

    var_names = top_up["feature"].tolist() + top_down["feature"].tolist()

    # Close any preexisting figures
    plt.close("all")

    # Call the plotting function
    fig = dx.pl.matrixplot(
        adata,
        var_names=var_names,
        groupby=["condition", "cell_type"],
        layer="binary",
        cmap="viridis",
        standard_scale="var",
        vmin=0,
        vmax=1,
        return_fig=True,  # required for testability
        show=False,  # prevent blocking
    )

    assert fig is not None
    assert isinstance(fig.fig, plt.Figure)
