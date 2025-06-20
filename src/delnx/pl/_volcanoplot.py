from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .._settings import settings

color_discrete_map = {
    "NS": "#d1d5db",  # gray-300
    "Up": "#ef4444",  # red-500
    "Down": "#3b82f6",  # blue-500
}


class VolcanoPlot:
    """Scanpy-style interactive volcano plot using Plotly."""

    DEFAULT_COLOR_LEGEND_TITLE = "-log10(p-value)"
    DEFAULT_SAVE_PREFIX = "volcanoplot_"

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
        self.fig: go.Figure | None = None

        if "-log10(pval)" not in self.df.columns:
            self.df["-log10(pval)"] = -np.log10(self.df["pval"])

    def style(self, color_map: dict[str, str] | None = None) -> VolcanoPlot:
        """Optional styling method (currently placeholder)."""
        if color_map is not None:
            global color_discrete_map
            color_discrete_map = color_map
        return self

    def make_figure(self) -> VolcanoPlot:
        """Creates the plotly volcano figure."""
        fig = px.scatter(
            self.df,
            x="coef",
            y="-log10(pval)",
            color="significant",
            hover_data=["feature"] if "feature" in self.df.columns else None,
            template="simple_white",
            color_discrete_map=color_discrete_map,
            category_orders={"significant": ["NS", "Up", "Down"]},
        )

        fig.add_hline(y=-np.log10(self.pval_thresh), opacity=1, line_width=1, line_dash="dash", line_color="black")
        fig.add_vline(x=self.coef_thresh, opacity=1, line_width=1, line_dash="dash", line_color="black")
        fig.add_vline(x=-self.coef_thresh, opacity=1, line_width=1, line_dash="dash", line_color="black")

        fig.update_layout(
            xaxis_title="Estimated Coefficient",
            yaxis_title="-log10(p-value)",
            legend_title=self.color_legend_title,
        )
        self.fig = fig
        return self

    def show(self):
        if self.fig is None:
            self.make_figure()
        self.fig.show()

    def save(self):
        if self.fig and self.save_path:
            self.fig.write_image(self.save_path)

    def get_figure(self) -> go.Figure:
        if self.fig is None:
            self.make_figure()
        return self.fig


def volcanoplot(
    df: pd.DataFrame,
    coef_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    color_legend_title: str | None = None,
    show: bool | None = None,
    save: str | bool | None = None,
    return_fig: bool = False,
) -> VolcanoPlot | go.Figure | None:
    """
    Volcano plot of coefficient vs. -log10(p-value), colored by significance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'coef', 'pval', 'significant', and optionally 'feature'.
    coef_thresh : float
        Threshold for coefficient cutoff (vertical lines).
    pval_thresh : float
        P-value threshold (horizontal line).
    color_legend_title : str or None
        Title for the legend. Defaults to "-log10(p-value)".
    show : bool or None
        Whether to show the figure interactively.
    save : str or bool or None
        If str, path to save the image. If True, saves with default filename.
    return_fig : bool
        Whether to return the Plotly figure object.

    Returns
    -------
    VolcanoPlot | plotly.graph_objs.Figure | None
    """
    save_path = None
    if isinstance(save, str):
        save_path = save
    elif save is True:
        save_path = f"{VolcanoPlot.DEFAULT_SAVE_PREFIX}plot.pdf"

    vp = VolcanoPlot(
        df,
        coef_thresh=coef_thresh,
        pval_thresh=pval_thresh,
        color_legend_title=color_legend_title,
        save_path=save_path,
    ).make_figure()

    if save_path:
        vp.save()

    show = settings.autoshow if show is None else show
    if show:
        vp.show()

    if return_fig:
        return vp.get_figure()

    return vp
