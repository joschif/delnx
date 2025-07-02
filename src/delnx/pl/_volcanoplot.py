import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from delnx.pp._get_de_genes import get_de_genes


class VolcanoPlot:
    """Static volcano plot using matplotlib."""

    DEFAULT_COLOR_LEGEND_TITLE = "-log10(p-value)"
    DEFAULT_SAVE_PREFIX = "volcanoplot"

    def __init__(
        self,
        df: pd.DataFrame,
        coef_thresh: float = 1.0,
        pval_thresh: float = 0.05,
        color_legend_title: str | None = None,
        save_path: str | None = None,
    ):
        self.df = df.copy()
        self.coef_thresh = coef_thresh
        self.pval_thresh = pval_thresh
        self.color_legend_title = color_legend_title or self.DEFAULT_COLOR_LEGEND_TITLE
        self.save_path = save_path
        self.fig = None
        self.ax = None

        if "-log10(pval)" not in self.df.columns:
            self.df["-log10(pval)"] = -np.log10(self.df["pval"])

        # Default color map
        self.color_map = {
            "NS": "#d1d5db",  # gray-300
            "Up": "#ef4444",  # red-500
            "Down": "#3b82f6",  # blue-500
        }

    def style(self, color_map: dict[str, str] | None = None) -> "VolcanoPlot":
        if color_map:
            self.color_map = color_map
        return self

    def make_figure(self) -> "VolcanoPlot":
        fig, ax = plt.subplots(figsize=(8, 6))
        self.fig = fig
        self.ax = ax

        # Plot each significance group
        for label, color in self.color_map.items():
            subset = self.df[self.df["significant"] == label]
            ax.scatter(
                subset["coef"],
                subset["-log10(pval)"],
                c=color,
                label=label,
                edgecolor="black",
                linewidth=0.5,
                s=20,
                alpha=0.8,
            )

        # Threshold lines
        ax.axhline(y=-np.log10(self.pval_thresh), color="black", linestyle="--", linewidth=1)
        ax.axvline(x=self.coef_thresh, color="black", linestyle="--", linewidth=1)
        ax.axvline(x=-self.coef_thresh, color="black", linestyle="--", linewidth=1)

        # Labels and grid
        ax.set_xlabel("Estimated Coefficient")
        ax.set_ylabel("-log10(p-value)")
        ax.legend(title=self.color_legend_title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        return self

    def add_labels(self, top_up: list[str], top_down: list[str]) -> None:
        if self.ax is None:
            raise RuntimeError("Plot must be initialized before adding labels.")
        texts = []
        for feature in top_up + top_down:
            row = self.df[self.df["feature"] == feature].iloc[0]
            texts.append(
                self.ax.text(
                    row["coef"],
                    row["-log10(pval)"],
                    feature,
                    fontsize=8,
                    ha="right" if row["coef"] > 0 else "left",
                    va="bottom",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": "black",
                        "linewidth": 0.5,
                    },
                )
            )
        adjust_text(
            texts,
            ax=self.ax,
            arrowprops={
                "arrowstyle": "-",
                "color": "gray",
                "lw": 0.5,
            },
        )

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


def volcanoplot(
    df: pd.DataFrame,
    coef_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    label_top: int = 0,
    color_legend_title: str | None = None,
    show: bool | None = True,
    save: str | bool | None = None,
    return_fig: bool = False,
):
    """
    Create a volcano plot using matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'coef', 'pval', 'significant', and optionally 'feature'.
    coef_thresh : float
        Coefficient threshold for vertical cutoff lines.
    pval_thresh : float
        P-value threshold for horizontal cutoff line.
    label_top : int
        If > 0, label top N up/down genes by effect size.
    color_legend_title : str or None
        Title for the legend. Default: "-log10(p-value)".
    show : bool or None
        Whether to display the figure interactively.
    save : str or bool or None
        If str, path to save the image. If True, uses default name.
    return_fig : bool
        Whether to return the matplotlib Figure and Axes.

    Returns
    -------
    VolcanoPlot or tuple[Figure, Axes] or None
    """
    # Check group uniqueness if group column exists
    if "group" in df.columns:
        unique_groups = df["group"].unique()
        if len(unique_groups) > 1:
            raise ValueError(f"Volcano plot expects a single group, but found multiple: {unique_groups}")

    save_path = None
    if isinstance(save, str):
        save_path = save
    elif save is True:
        save_path = f"{VolcanoPlot.DEFAULT_SAVE_PREFIX}.pdf"

    vp = VolcanoPlot(
        df,
        coef_thresh=coef_thresh,
        pval_thresh=pval_thresh,
        color_legend_title=color_legend_title,
        save_path=save_path,
    ).make_figure()

    if label_top > 0 and "feature" in df.columns:
        de_genes_dict = get_de_genes(df, top_n=label_top)
        de_genes_dict = de_genes_dict[list(de_genes_dict.keys())[0]]
        top_up, top_down = de_genes_dict["up"], de_genes_dict["down"]
        vp.add_labels(top_up, top_down)

    if save_path:
        vp.save()
    if show:
        vp.show()
    if return_fig:
        return vp.get_figure()
    return vp
