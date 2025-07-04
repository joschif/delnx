import matplotlib.pyplot as plt
import numpy as np


def plot_filtered_genes(adata, keep_mask=None, layer=None):
    """
    Plot ranked total counts per gene from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Single-cell AnnData object.
    keep_mask : np.ndarray or None
        Boolean array of shape (n_genes,) indicating which genes are retained.
        If provided, retained and filtered genes will be shown in different colors.
    layer : str or None
        If provided, use `adata.layers[layer]` instead of `adata.X`.
    """
    # Get count matrix
    if layer is not None:
        counts = adata.layers[layer]
    else:
        counts = adata.X

    # Compute total counts per gene
    if isinstance(counts, np.ndarray):
        total_counts = counts.sum(axis=0)
    else:
        total_counts = np.array(counts.sum(axis=0)).ravel()

    # Sort by decreasing total counts
    sorted_idx = np.argsort(-total_counts)
    sorted_counts = total_counts[sorted_idx]

    # Plot
    plt.figure(figsize=(10, 4))
    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask)[sorted_idx]
        plt.scatter(
            np.arange(len(sorted_counts))[keep_mask], sorted_counts[keep_mask], color="blue", s=2, label="kept genes"
        )
        plt.scatter(
            np.arange(len(sorted_counts))[~keep_mask],
            sorted_counts[~keep_mask],
            color="lightgray",
            s=2,
            label="filtered out",
        )
        plt.legend()
    else:
        plt.plot(sorted_counts, lw=1)

    plt.yscale("log")
    plt.xlabel("Gene rank (by total count)")
    plt.ylabel("Total counts per gene (log scale)")
    plt.title("Gene count distribution")
    plt.tight_layout()
    plt.show()
