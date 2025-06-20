import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class GeneSetEnrichmentPlot:
    """Gene set enrichment barplot using matplotlib."""

    DEFAULT_SAVE_PREFIX = "geneset_enrichment_"

    def __init__(
        self,
        genes: list[str],
        background: list[str] | None = None,
        organism: str = "Human",
        method: str = "enrichr",
        library: str = "GO_Biological_Process_2021",
        top_n: int = 10,
        save_path: str | None = None,
    ):
        self.genes = genes
        self.background = background
        self.organism = organism
        self.method = method
        self.library = library
        self.top_n = top_n
        self.save_path = save_path
        self.results: pd.DataFrame | None = None
        self.fig = None
        self.ax = None

    def run(self):
        if self.method == "enrichr":
            enr = gp.enrichr(
                gene_list=self.genes,
                background=self.background,
                organism=self.organism,
                gene_sets=self.library,
                outdir=None,
                no_plot=True,
            )
            self.results = enr.results
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        return self

    def make_figure(self) -> "GeneSetEnrichmentPlot":
        if self.results is None:
            raise RuntimeError("You must run .run() before plotting.")

        top_terms = self.results.sort_values("Adjusted P-value").head(self.top_n)

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * self.top_n)))
        self.fig = fig
        self.ax = ax

        ax.barh(
            y=top_terms["Term"],
            width=-np.log10(top_terms["Adjusted P-value"]),
            color="#3b82f6",
            edgecolor="black",
        )

        ax.set_xlabel("-log10(Adjusted P-value)")
        ax.set_ylabel("Enriched Term")
        ax.invert_yaxis()  # Most significant on top
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        return self

    def show(self):
        if self.fig is None:
            self.make_figure()
        plt.show()

    def save(self):
        if self.fig and self.save_path:
            self.fig.savefig(self.save_path, bbox_inches="tight", dpi=300)

    def get_figure(self):
        if self.fig is None:
            self.make_figure()
        return self.fig, self.ax


def geneset_enrichment_plot(
    gene_set: list[str],
    background: list[str] | None = None,
    organism: str = "Human",
    library: str = "GO_Biological_Process_2021",
    method: str = "enrichr",
    top_n: int = 10,
    show: bool = True,
    save: str | bool | None = None,
    return_fig: bool = False,
):
    """
    Perform gene set enrichment and plot top enriched terms.

    Parameters
    ----------
    gene_set : list of str
        List of gene symbols to test.
    background : list of str, optional
        Background gene list (if desired).
    organism : str
        Organism for Enrichr (e.g. "Human", "Mouse").
    library : str
        Enrichr gene set library (e.g. "GO_Biological_Process_2021").
    method : str
        Enrichment method, only 'enrichr' is supported.
    top_n : int
        Number of top terms to show.
    show : bool
        Whether to display the plot.
    save : str | bool | None
        File path or True to save with default name.
    return_fig : bool
        Return matplotlib figure and axes.

    Returns
    -------
    GeneSetEnrichmentPlot or tuple[Figure, Axes] or None
    """
    save_path = None
    if isinstance(save, str):
        save_path = save
    elif save is True:
        save_path = f"{GeneSetEnrichmentPlot.DEFAULT_SAVE_PREFIX}plot.pdf"

    plotter = GeneSetEnrichmentPlot(
        genes=gene_set,
        background=background,
        organism=organism,
        method=method,
        library=library,
        top_n=top_n,
        save_path=save_path,
    )
    plotter.run().make_figure()

    if save_path:
        plotter.save()
    if show:
        plotter.show()
    if return_fig:
        return plotter.get_figure()
    return plotter
