import gseapy as gp
import pandas as pd

def run_gsea(
    genes: list[str],
    background: list[str] | None = None,
    organism: str = "Human",
    method: str = "enrichr",
    library: str = "GO_Biological_Process_2021",
) -> pd.DataFrame:
    """
    Compute gene set enrichment using gseapy.

    Returns
    -------
    pd.DataFrame
        Enrichment results table.
    """
    if method == "enrichr":
        enr = gp.enrichr(
            gene_list=genes,
            background=background,
            organism=organism,
            gene_sets=library,
            outdir=None,
            no_plot=True,
        )
        return enr.results
    else:
        raise ValueError(f"Unsupported method: {method}")
