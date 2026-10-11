import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

from delnx._utils import get_de_genes


class VolcanoPlot:
    """Static volcano plot using matplotlib."""

    DEFAULT_COLOR_LEGEND_TITLE = "-log10(p-value)"
    DEFAULT_SAVE_PREFIX = "volcanoplot"

    def __init__(
        self,
        df: pd.DataFrame,
        x: str = "coef",
        y: str = "-log10(pval)",
        thresh: dict[str, float] | None = None,
        color_legend_title: str | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (8, 6),
        save_path: str | None = None,
    ):
        self.df = df.copy()
        self.x = x
        self.y = y
        self.thresh = thresh or {}
        self.color_legend_title = color_legend_title or self.DEFAULT_COLOR_LEGEND_TITLE
        self.save_path = save_path
        self.figsize = figsize
        self.ax = ax

        # Compute -log10(pval) if needed and y is set to that
        if self.y == "-log10(pval)" and "-log10(pval)" not in self.df.columns:
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
        if self.ax is not None:
            self.fig = self.ax.figure
        else:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Plot each significance group
        for label, color in self.color_map.items():
            subset = self.df[self.df["significant"] == label]
            self.ax.scatter(
                subset[self.x],
                subset[self.y],
                c=color,
                label=label,
                edgecolor="black",
                linewidth=0.5,
                s=20,
                alpha=0.8,
            )

        # Threshold lines
        x_thresh = self.thresh.get(self.x, None)
        y_thresh = self.thresh.get(self.y, None)
        if y_thresh is not None:
            self.ax.axhline(y=y_thresh, color="black", linestyle="--", linewidth=1)
        if x_thresh is not None:
            self.ax.axvline(x=x_thresh, color="black", linestyle="--", linewidth=1)
            self.ax.axvline(x=-x_thresh, color="black", linestyle="--", linewidth=1)

        # Labels and grid
        self.ax.set_xlabel(self.x if self.x != "coef" else "Estimated Coefficient")
        self.ax.set_ylabel(self.y if self.y != "-log10(pval)" else "-log10(p-value)")
        self.ax.legend(title=self.color_legend_title)
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        return self

    def add_labels(self, top_up: list[str], top_down: list[str]) -> None:
        if self.ax is None:
            raise RuntimeError("Plot must be initialized before adding labels.")
        texts = []
        for feature in top_up + top_down:
            row = self.df[self.df["feature"] == feature].iloc[0]
            texts.append(
                self.ax.text(
                    row[self.x],
                    row[self.y],
                    feature,
                    fontsize=8,
                    ha="right" if row[self.x] > 0 else "left",
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
    x: str = "log2fc",
    y: str = "-log10(pval)",
    effect_key: str = "log2fc",
    pval_key: str = "pval",
    feature_key: str = "feature",
    effect_thresh: float = 0.5,
    pval_thresh: float = 0.01,
    thresh: dict[str, float] | None = None,
    label_top: int = 0,
    color_legend_title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    show: bool | None = True,
    save: str | bool | None = None,
    return_fig: bool = False,
):
    """
    Create a volcano plot using matplotlib. Labels significant genes based on thresholds or uses "significant" column if present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing differential expression results.
    x : str, default="log2fc"
        Column name for x-axis (effect size).
    y : str, default="-log10(pval)"
        Column name for y-axis (significance).
    effect_key : str, default="log2fc"
        Column with effect size values for DE analysis.
    pval_key : str, default="pval"
        Column with p-values for DE analysis.
    feature_key : str, default="feature"
        Column containing gene names for labeling.
    effect_thresh : float, default=0.5
        Threshold for absolute effect size.
    pval_thresh : float, default=0.01
        Threshold for significance.
    thresh : dict[str, float] or None
        Dictionary mapping axis names to threshold values for plot lines,
        e.g. {'log2fc': 1.0, '-log10(pval)': 1.3}.
        If None, uses effect_thresh and -log10(pval_thresh).
    label_top : int, default=0
        If > 0, label top N up/down genes by effect size.
    color_legend_title : str or None
        Title for the legend. Default: "-log10(p-value)".
    ax : plt.Axes or None
        If provided, use this Axes for plotting instead of creating a new one.
        If None, a new Axes will be created.
    figsize : tuple[float, float], default=(8, 6)
        Size of the figure in inches.
    show : bool or None, default=True
        Whether to display the figure interactively.
    save : str or bool or None
        If str, path to save the image. If True, uses default name.
    return_fig : bool, default=False
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

    # Use analyze_de_genes to label the dataframe if needed
    df_to_plot = df.copy()

    # Check if significance labels already exist, if not, add them
    if "significant" not in df_to_plot.columns or y not in df_to_plot.columns:
        _, df_to_plot = get_de_genes(
            df,
            effect_key=effect_key,
            pval_key=pval_key,
            feature_key=feature_key,
            effect_thresh=effect_thresh,
            pval_thresh=pval_thresh,
            return_labeled_df=True,
        )

    # Set default thresholds if not provided
    if thresh is None:
        thresh = {x: effect_thresh, y: -np.log10(pval_thresh)}

    save_path = None
    if isinstance(save, str):
        save_path = save
    elif save is True:
        save_path = f"{VolcanoPlot.DEFAULT_SAVE_PREFIX}.pdf"

    vp = VolcanoPlot(
        df_to_plot,
        x=x,
        y=y,
        thresh=thresh,
        color_legend_title=color_legend_title,
        figsize=figsize,
        ax=ax,
        save_path=save_path,
    ).make_figure()

    # Add labels for top genes if requested
    if label_top > 0 and feature_key in df_to_plot.columns:
        # Extract top genes using our unified function
        de_genes_dict = get_de_genes(
            df_to_plot,
            effect_key=effect_key,
            pval_key=pval_key,
            feature_key=feature_key,
            effect_thresh=effect_thresh,
            pval_thresh=pval_thresh,
            top_n=label_top,
        )

        # Get the first (and should be only) group's genes
        group_key = list(de_genes_dict.keys())[0]
        top_up = de_genes_dict[group_key]["up"]
        top_down = de_genes_dict[group_key]["down"]

        vp.add_labels(top_up, top_down)

    if save_path:
        vp.save()
    if show:
        vp.show()
    if return_fig:
        return vp.get_figure()
