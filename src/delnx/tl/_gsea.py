from typing import Any

import gseapy as gp
import pandas as pd

from ..pp._get_de_genes import get_de_genes


def run_single_enrichment_analysis(
    genes: list[str],
    background: list[str] | None = None,
    organism: str = "Human",
    method: str = "enrichr",
    library: str = "GO_Biological_Process_2021",
    return_object: bool = False,
) -> pd.DataFrame | Any:
    """
    Run enrichment for a single gene list.

    Parameters
    ----------
    ...
    return_object : bool
        If True, return the enrichment object. Otherwise, return results table.

    Returns
    -------
    Union[pd.DataFrame, object]
    """
    if method != "enrichr":
        raise ValueError(f"Unsupported method: {method}")

    enr = gp.enrichr(
        gene_list=genes,
        background=background,
        organism=organism,
        gene_sets=library,
        outdir=None,
        no_plot=True,
    )

    return enr if return_object else enr.res2d


def run_de_enrichment_analysis(
    de_results: pd.DataFrame,
    top_n: int = 200,
    background: list[str] | None = None,
    organism: str = "Human",
    method: str = "enrichr",
    library: str = "GO_Biological_Process_2021",
    cutoff: float = 0.05,
) -> pd.DataFrame:
    """
    Run enrichment for up/down gene sets per group and stack filtered results.

    Parameters
    ----------
    de_results : pd.DataFrame
        DataFrame with differential expression results, must contain group labels.
    top_n : int
        Number of top genes to select for each group and direction.
    background : list or None
        Optional background gene list.
    organism : str
        Organism name for enrichment (e.g., "Human").
    method : str
        Enrichment method (currently supports "enrichr").
    library : str
        Enrichment gene set library.
    cutoff : float
        Adjusted p-value cutoff for significance.

    Returns
    -------
    pd.DataFrame
        Combined enrichment results across groups.
    """
    de_genes_dict = get_de_genes(de_results, top_n=top_n)
    all_enrichment_results = []

    for group, gene_sets in de_genes_dict.items():
        up_genes = gene_sets.get("up", [])
        down_genes = gene_sets.get("down", [])

        # Run enrichment only if gene list is not empty
        if up_genes:
            enr_up = run_single_enrichment_analysis(up_genes, background, organism, method, library)
            enr_up = enr_up[enr_up["Adjusted P-value"] <= cutoff].copy()
            enr_up["UP_DW"] = "UP"
            enr_up["group"] = group
            all_enrichment_results.append(enr_up)

        if down_genes:
            enr_dw = run_single_enrichment_analysis(down_genes, background, organism, method, library)
            enr_dw = enr_dw[enr_dw["Adjusted P-value"] <= cutoff].copy()
            enr_dw["UP_DW"] = "DOWN"
            enr_dw["group"] = group
            all_enrichment_results.append(enr_dw)

    if all_enrichment_results:
        return pd.concat(all_enrichment_results, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if nothing passes cutoff
