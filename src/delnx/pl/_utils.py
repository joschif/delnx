from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt

# -------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------


def savefig(writekey, dpi=150, ext="png", figdir=".", plot_suffix=""):
    """
    Save the current matplotlib figure to a file.

    Parameters
    ----------
    writekey : str
        Base filename (without extension).
    dpi : int
        Resolution of saved figure.
    ext : str
        File format (e.g., 'png', 'pdf').
    figdir : str or Path
        Directory to save the figure.
    plot_suffix : str
        Suffix to append to filename.
    """
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    filename = figdir / f"{writekey}{plot_suffix}.{ext}"
    print(f"saving figure to file {filename}")
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")


def save_or_show(
    writekey=None,
    show=True,
    save=True,
    dpi=150,
    ext="png",
    figdir=".",
    plot_suffix="",
):
    """
    Save or show the current matplotlib figure.

    Parameters
    ----------
    writekey : str or None
        Base filename (without extension).
    show : bool
        Whether to show the plot via `plt.show()`.
    save : bool or str
        If True, save using `writekey`.
        If str, treat as custom filename (can include extension).
    dpi : int
        Resolution of saved figure.
    ext : str
        File format (e.g., 'png', 'pdf').
    figdir : str or Path
        Directory to save the figure.
    plot_suffix : str
        Suffix to append to filename.
    """
    if isinstance(save, str):
        # extract extension if present
        if ext is None or ext == "":
            for try_ext in [".svg", ".pdf", ".png"]:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    save = save.replace(try_ext, "")
                    break
        writekey = save
        save = True

    if save and writekey:
        savefig(writekey, dpi=dpi, ext=ext, figdir=figdir, plot_suffix=plot_suffix)
    if show:
        plt.show()
    if save:
        plt.close()
