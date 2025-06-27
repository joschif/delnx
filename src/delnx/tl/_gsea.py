from collections.abc import Sequence
from typing import Any

import gseapy as gp
import pandas as pd

from ..ds._gmt import load_gmt
from ..pp._get_de_genes import get_de_genes

MIN_GENESET_SIZE = 5
MAX_GENESET_SIZE = 500


def get_gene_sets(
    collection: str = "all",
    url: str | None = None,
    filepath: str | None = None,
    geneset_key: str = "geneset",
    genesymbol_key: str = "genesymbol",
    min_genes: int = MIN_GENESET_SIZE,
    max_genes: int = MAX_GENESET_SIZE,
) -> dict[str, list[str]]:
    """
    Load and return gene sets as a dictionary.
    """
    gmt_df = load_gmt(
        collection=collection,
        url=url,
        filepath=filepath,
        geneset_key=geneset_key,
        genesymbol_key=genesymbol_key,
        min_genes=min_genes,
        max_genes=max_genes,
    )
    gmt_df = gmt_df.rename(columns={geneset_key: "source", genesymbol_key: "target"})
    gene_sets = gmt_df.groupby("source")["target"].apply(list).to_dict()
    return gene_sets


def single_enrichment_analysis(
    genes: Sequence[str],
    background: Sequence[str] | None = None,
    gene_sets: dict[str, list[str]] | None = None,
    collection: str = "all",
    url: str | None = None,
    filepath: str | None = None,
    geneset_key: str = "geneset",
    genesymbol_key: str = "genesymbol",
    method: str = "enrichr",
    return_object: bool = False,
    min_genes: int = MIN_GENESET_SIZE,
    max_genes: int = MAX_GENESET_SIZE,
) -> pd.DataFrame | Any:
    """
    Run enrichment analysis for a single gene list using Enrichr.
    """
    if method != "enrichr":
        raise ValueError(f"Unsupported method: {method}")

    if gene_sets is None:
        gene_sets = get_gene_sets(
            collection=collection,
            url=url,
            filepath=filepath,
            geneset_key=geneset_key,
            genesymbol_key=genesymbol_key,
            min_genes=min_genes,
            max_genes=max_genes,
        )

    enr = gp.enrichr(
        gene_list=list(genes),
        background=list(background) if background is not None else None,
        gene_sets=gene_sets,
        outdir=None,
        no_plot=True,
    )

    return enr if return_object else enr.res2d


def de_enrichment_analysis(
    de_results: pd.DataFrame,
    top_n: int | None = None,
    background: Sequence[str] | None = None,
    collection: str = "all",
    url: str | None = None,
    filepath: str | None = None,
    geneset_key: str = "geneset",
    genesymbol_key: str = "genesymbol",
    method: str = "enrichr",
    cutoff: float = 0.05,
    min_genes: int = MIN_GENESET_SIZE,
    max_genes: int = MAX_GENESET_SIZE,
) -> pd.DataFrame:
    """
    Run enrichment for up/down gene sets per group and stack filtered results.
    """
    de_genes_dict = get_de_genes(de_results, top_n=top_n)
    all_enrichment_results = []

    # Load gene sets once and reuse
    gene_sets = get_gene_sets(
        collection=collection,
        url=url,
        filepath=filepath,
        geneset_key=geneset_key,
        genesymbol_key=genesymbol_key,
        min_genes=min_genes,
        max_genes=max_genes,
    )

    for group, gene_sets_dict in de_genes_dict.items():
        for direction, label in [("up", "UP"), ("down", "DOWN")]:
            genes = gene_sets_dict.get(direction, [])
            if not genes:
                continue
            enr = single_enrichment_analysis(
                genes,
                background=background,
                gene_sets=gene_sets,
                method=method,
            )
            if "Adjusted P-value" in enr.columns:
                filtered = enr[enr["Adjusted P-value"] <= cutoff].copy()
                filtered["UP_DW"] = label
                filtered["group"] = group
                all_enrichment_results.append(filtered)

    if all_enrichment_results:
        return pd.concat(all_enrichment_results, ignore_index=True)
    return pd.DataFrame()
