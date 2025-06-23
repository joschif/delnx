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
    """Barplot showing top UP and DOWN enriched terms separately."""
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
    x_order: list[str] = ["UP", "DOWN"],
    top_n: int = 10,
    title: str = "GO_BP",
    cmap: str = "Blues",
    figsize=(4, 5),
    cutoff: float = 0.05,
    show_all: bool = False,
) -> plt.Axes:
    """
    Plot dotplot for enrichment results using GSEApy.
    """
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
        raise ValueError(f"No enriched terms to plot (cutoff = {cutoff}): {e}")
