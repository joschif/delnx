import decoupler as dc
import numpy as np
from anndata import AnnData


def pseudobulk(
    adata: AnnData,
    sample_key: str = "batch",
    group_key: str | None = None,
    n_pseudoreps: int | None = None,
    layer: str | None = None,
    mode: str = "sum",
    min_cells: int = 10,
    min_counts: int = 1000,
    **kwargs,
) -> AnnData:
    """
    Create pseudobulk data from anndata object

    Parameters
    ----------
    adata : AnnData
        AnnData object
    sample_key : str, default="batch"
        Column name in adata.obs that contains the sample ID
    group_key : str or None, default=None
        Column name in adata.obs that contains the group ID
    n_pseudoreps : int or None, default=None
        Number of pseudoreplicates to create. If None, use the original samples.
        If specified, will create `n_pseudoreps` pseudoreplicates per sample.
    layer : str or None, default=None
        Layer to use for pseudobulk data. If None, use `adata.X`
    mode : str, default="sum"
        Method to aggregate data
    min_cells : int, default=10
        Filter to remove samples by a minimum number of cells in a sample-group pair
    min_counts : int, default=1000
        Filter to remove samples by a minimum number of summed counts in a sample-group pair
    **kwargs
        Additional arguments to pass to `decoupler.get_pseudobulk()`

    Requires
    --------
    The decoupler package must be installed:
    `pip install decoupler`

    Returns
    -------
    AnnData
        AnnData object with pseudobulk data
    """
    if n_pseudoreps is not None:
        pseudoreps = np.random.choice(
            np.arange(n_pseudoreps),
            size=adata.n_obs,
            replace=True,
        )
        adata.obs["psbulk_replicate"] = adata.obs[sample_key] + "_" + pseudoreps.astype(str)
    else:
        adata.obs["psbulk_replicate"] = adata.obs[sample_key]

    return dc.pp.pseudobulk(
        adata,
        sample_col="psbulk_replicate",
        groups_col=group_key,
        layer=layer,
        mode=mode,
        min_cells=min_cells,
        min_counts=min_counts,
        **kwargs,
    )
