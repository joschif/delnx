from collections.abc import Sequence
from typing import Any

import gseapy as gp
import pandas as pd

from delnx._utils import get_de_genes
from delnx.ds._gmt import get_gene_sets

MIN_GENESET_SIZE = 5
MAX_GENESET_SIZE = 500


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

    Parameters
    ----------
    genes : Sequence[str]
        List of gene symbols to analyze.
    background : Sequence[str], optional
        Background gene list to use for enrichment analysis. If None, uses all genes in the gene sets.
    gene_sets : dict[str, list[str]], optional
        Pre-loaded gene sets as a dictionary where keys are gene set names and values are lists of gene symbols.
        If None, will load gene sets based on the provided collection, URL, or filepath.
    collection : str
        Name of the collection to load gene sets from. Default is "all".
    url : str, optional
        URL to load the GMT file from. If None, uses the default collection.
    filepath : str, optional
        Local file path to load the GMT file from. If None, uses the default collection.
    geneset_key : str
        Column name for the gene set name in the output dictionary. Default is "geneset".
    genesymbol_key : str
        Column name for the gene symbol in the output dictionary. Default is "genesymbol".
    method : str
        Method to use for enrichment analysis. Currently only "enrichr" is supported.
    return_object : bool
        If True, returns the gseapy Enrichr object. If False, returns a pandas DataFrame with results.
    min_genes : int
        Minimum number of genes in a gene set to include in the analysis. Default is 5.
    max_genes : int
        Maximum number of genes in a gene set to include in the analysis. Default is 500.
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
    group_col: str = "group",
    effect_key: str = "log2fc",
    pval_key: str = "padj",
    gene_col: str = "feature",
    effect_thresh: float = 0.5,
    pval_thresh: float = 0.01,
    gene_sets: dict[str, list[str]] | None = None,
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
    Uses get_de_genes to extract significant genes based on thresholds.

    Parameters
    ----------
    de_results : pd.DataFrame
        DataFrame containing differential expression results.
    group_col : str, default="group"
        Column indicating the group or condition.
    effect_key : str, default="log2fc"
        Column with effect size values for DE analysis.
    pval_key : str, default="pval"
        Column with p-values for DE analysis.
    gene_col : str, default="feature"
        Column containing gene names.
    effect_thresh : float, default=0.5
        Threshold for absolute effect size. Threshold for significance of DE genes.
    pval_thresh : float, default=0.01
        Threshold for significance of DE genes.
    gene_sets : dict[str, list[str]] | None
        Pre-loaded gene sets as a dictionary where keys are gene set names and values are lists of gene symbols.
        If None, will load gene sets based on the provided collection, URL, or filepath.
    top_n : int, optional
        If specified, only consider the top N genes per group for enrichment analysis.
    background : Sequence[str], optional
        Background gene list to use for enrichment analysis. If None, uses all genes in the gene sets.
    collection : str, default="all"
        Name of the collection to load gene sets from.
    url : str, optional
        URL to load the GMT file from. If None, uses the default collection.
    filepath : str, optional
        Local file path to load the GMT file from. If None, uses the default collection.
    geneset_key : str, default="geneset"
        Column name for the gene set name in the output dictionary.
    genesymbol_key : str, default="genesymbol"
        Column name for the gene symbol in the output dictionary.
    method : str, default="enrichr"
        Method to use for enrichment analysis. Currently only "enrichr" is supported.
    cutoff : float, default=0.05
        Adjusted p-value cutoff for filtering enrichment results.
    min_genes : int, default=MIN_GENESET_SIZE
        Minimum number of genes in a gene set to include in the analysis.
    max_genes : int, default=MAX_GENESET_SIZE
        Maximum number of genes in a gene set to include in the analysis.

    Returns
    -------
    pd.DataFrame
        DataFrame containing enrichment results for each group and direction, filtered by the specified cutoff.
        Columns include 'geneset', 'genesymbol', 'Adjusted P-value', 'up_dw', and 'group'.

    Examples
    --------
    >>> de_results = pd.DataFrame(
    ...     {
    ...         "group": ["A", "A", "B", "B"],
    ...         "log2fc": [1.5, -2.0, 0.8, -1.2],
    ...         "pval": [0.01, 0.02, 0.03, 0.04],
    ...         "feature": ["gene1", "gene2", "gene3", "gene4"],
    ...     }
    ... )
    >>> enrichment_results = de_enrichment_analysis(de_results, top_n=10)
    >>> print(enrichment_results.head())
    """  # noqa: D205
    de_genes_dict = get_de_genes(
        de_results,
        group_col=group_col,
        effect_key=effect_key,
        pval_key=pval_key,
        gene_col=gene_col,
        effect_thresh=effect_thresh,
        pval_thresh=pval_thresh,
        top_n=top_n,
    )

    all_enrichment_results = []

    if gene_sets is None:
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

            if enr is None or enr.empty:
                continue

            if "Adjusted P-value" in enr.columns:
                filtered = enr[enr["Adjusted P-value"] <= cutoff].copy()
                filtered["up_dw"] = label
                filtered["group"] = group
                all_enrichment_results.append(filtered)

    if all_enrichment_results:
        return pd.concat(all_enrichment_results, ignore_index=True)
    return pd.DataFrame()
