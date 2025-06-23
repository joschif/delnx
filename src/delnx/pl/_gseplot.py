import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_gsea(
    enrichment_results: pd.DataFrame,
    top_n: int = 10,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot top enriched terms from enrichment results.

    Returns
    -------
    (Figure, Axes)
    """
    top_terms = enrichment_results.sort_values("Adjusted P-value").head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * top_n)))
    ax.barh(
        y=top_terms["Term"],
        width=-np.log10(top_terms["Adjusted P-value"]),
        color="#3b82f6",
        edgecolor="black",
    )
    ax.set_xlabel("-log10(Adjusted P-value)")
    ax.set_ylabel("Enriched Term")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    return fig, ax
