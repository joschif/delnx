from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from delnx.pl._baseplot import BasePlot


from dataclasses import dataclass
import numpy as np
import pandas as pd
import marsilea as ma
import marsilea.plotter as mp
import matplotlib.pyplot as plt

@dataclass
class GradientHeatmapPlot(BasePlot):
    """
    Heatmap class for plotting cells sorted by a continuous variable (e.g., pseudotime)
    with genes ordered by peak activation time and a continuous gradient annotation.

    Parameters
    ----------
    trajectory_key : str
        The name of the continuous variable in `adata.obs` to sort cells by.
    smooth : bool, default=True
        Whether to smooth gene expression along the x-axis (e.g., for pseudotime).
    smoothing_window : int, default=5
        Rolling average window size used if smoothing is enabled.
    gradient_cmap : str, default="Reds"
        Colormap used for the continuous gradient.
    """

    trajectory_key: str = "pseudotime"
    smooth: bool = True
    smoothing_window: int = 5
    gradient_cmap: str = "Reds"

    def _build_data(self) -> np.ndarray:
        """Extract and order expression matrix by trajectory and gene peak."""
        if self.trajectory_key not in self.adata.obs:
            raise ValueError(f"Trajectory key '{self.trajectory_key}' not found in adata.obs.")

        # Sort cells by the trajectory key
        traj = self.adata.obs[self.trajectory_key].astype(float)
        cell_order = np.argsort(traj.values)
        sorted_traj = traj.values[cell_order]

        # Extract expression matrix
        X = self.adata[cell_order, self.markers].X
        X = X.toarray() if not isinstance(X, np.ndarray) else X

        # Smooth expression if requested
        if self.smooth:
            for i in range(X.shape[1]):
                X[:, i] = pd.Series(X[:, i]).rolling(
                    window=self.smoothing_window,
                    min_periods=1,
                    center=True
                ).mean().values

        # Sort genes by peak activation (earliest peak first)
        peak_indices = X.argmax(axis=0)
        gene_order = np.argsort(peak_indices)
        self.markers = [self.markers[i] for i in gene_order]
        X = X[:, gene_order].T  # shape: genes x cells

        # Disable row grouping
        self.row_group = None
        self.order = None

        return X

    def _add_continuous_gradient(self, m: ma.Heatmap):
        """Add a continuous gradient annotation (e.g., pseudotime bar) above the heatmap."""
        traj = self.adata.obs[self.trajectory_key].astype(float)
        sorted_traj = traj.iloc[np.argsort(traj.values)].values

        # Normalize values between 0 and 1
        normed = (sorted_traj - sorted_traj.min()) / (sorted_traj.max() - sorted_traj.min())
        normed = normed.reshape(1, -1)  # shape (1, n_cells)

        ax = m.canvas.add_top(size=0.15, pad=0.02)
        ax.set_axis_off()
        ax.pcolormesh(normed, cmap=self.gradient_cmap, shading="nearest")
        ax.invert_yaxis()

    def _build_plot(self):
        """Build the heatmap plot."""
        data = self._build_data()

        # Create the Marsilea heatmap
        m = ma.Heatmap(
            data,
            cmap=self.cmap,
            height=self.height,
            width=self.width,
            vmin=self.vmin,
            vmax=self.vmax,
            center=self.center,
            cbar_kws={"title": "Expression"},
        )

        # Add pseudotime gradient
        self._add_continuous_gradient(m)

        # Add labels
        if self.show_column_names:
            self._add_column_labels(m)
        if self.show_row_names:
            self._add_row_labels(m)

        # Dendrograms (if enabled)
        if self.dendrograms:
            for pos in self.dendrograms:
                m.add_dendrogram(pos, add_base=False)

        # Legends (optional)
        if self.show_legends:
            m.add_legends()

        return m
