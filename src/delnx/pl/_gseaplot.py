import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gsea_barplot(
    enrichment_results: pd.DataFrame,
    top_n: int = 10,
    figsize=(4, 5),
    save_path: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a horizontal bar plot of top GSEA enrichment terms, split by UP and DOWN regulation.

    This function selects the top N significantly enriched terms separately for UP- and DOWN-regulated categories,
    based on the adjusted p-value. The selected terms are visualized as horizontal bars colored by regulation status.

    Parameters
    ----------
    enrichment_results : pd.DataFrame
        DataFrame containing GSEA results with at least the columns: "Term", "Adjusted P-value", and "UP_DW"
        (which should contain either "UP" or "DOWN" for each term).
    top_n : int, default=10
        Number of top enriched terms to include for each of the UP and DOWN categories.
    figsize : tuple, default=(4, 5)
        Size of the matplotlib figure (width, height in inches).
    save_path : str or None, default=None
        If provided, the plot will be saved to the given file path.
    show : bool, default=True
        If True, the plot will be displayed using `plt.show()`.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        A tuple containing the matplotlib figure and axes objects of the generated plot.
    """
    df = enrichment_results.copy()

    top_up = df[df["UP_DW"] == "UP"].sort_values("Adjusted P-value").head(top_n)
    top_down = df[df["UP_DW"] == "DOWN"].sort_values("Adjusted P-value").head(top_n)

    top_terms = pd.concat([top_up, top_down])
    top_terms["Label"] = top_terms["UP_DW"] + " | " + top_terms["Term"]

    fig, ax = plt.subplots(figsize=figsize)
    colors = top_terms["UP_DW"].map({"UP": "#10b981", "DOWN": "#ef4444"})

    ax.barh(
        y=top_terms["Label"],
        width=-np.log10(top_terms["Adjusted P-value"]),
        color=colors,
        edgecolor="black",
    )
    ax.set_xlabel("-log10(Adjusted P-value)")
    ax.set_ylabel("Enriched Terms (UP / DOWN)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    return fig, ax


def gsea_dotplot(
    enrichment_results: pd.DataFrame,
    x_order: list[str] | None = None,
    top_n: int = 10,
    title: str = "GO_BP",
    cmap: str = "Blues",
    figsize=(4, 5),
    cutoff: float = 0.05,
    show_all: bool = False,
) -> plt.Axes:
    """
    Create a dot plot visualization of top GSEA enrichment terms using GSEApy.

    This function selects the top N enriched terms from both UP- and DOWN-regulated categories
    based on adjusted p-values, and visualizes them using a dot plot where the x-axis reflects
    the direction of regulation and dot size/intensity encodes enrichment statistics.

    Parameters
    ----------
    enrichment_results : pd.DataFrame
        DataFrame containing GSEA results with at least the columns: "Term", "Adjusted P-value", and "UP_DW"
        (with values "UP" or "DOWN" indicating the direction of enrichment).
    x_order : list of str or None, default=["UP", "DOWN"]
        Order of categories to display along the x-axis. If None, defaults to ["UP", "DOWN"].
    top_n : int, default=10
        Number of top terms to include from each category (UP and DOWN).
    title : str, default="GO_BP"
        Title of the plot.
    cmap : str, default="Blues"
        Colormap to use for dot color intensity.
    figsize : tuple, default=(4, 5)
        Size of the matplotlib figure (width, height in inches).
    cutoff : float, default=0.05
        Adjusted p-value cutoff for filtering enriched terms. (Currently only used in error message.)
    show_all : bool, default=False
        Reserved for future use. Currently has no effect.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object of the generated plot.

    Raises
    ------
    ValueError
        If no enriched terms are available for plotting.
    """
    if x_order is None:
        x_order = ["UP", "DOWN"]

    df = enrichment_results.copy()

    top_up = df[df["UP_DW"] == "UP"].sort_values("Adjusted P-value").head(top_n)
    top_down = df[df["UP_DW"] == "DOWN"].sort_values("Adjusted P-value").head(top_n)

    top_terms = pd.concat([top_up, top_down])

    try:
        ax = gp.dotplot(
            top_terms,
            x="UP_DW",
            x_order=x_order,
            figsize=figsize,
            title=title,
            cmap=cmap,
            size=3,
            show_ring=True,
            cutoff=1.0,
        )
        ax.set_xlabel("")
        plt.show()
        return ax
    except ValueError as e:
        raise ValueError(f"No enriched terms to plot (cutoff = {cutoff}): {e}") from e
