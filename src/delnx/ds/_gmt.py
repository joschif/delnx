import os
from io import StringIO
from typing import Any

import pandas as pd
import requests

# Dictionary of possible MSigDB GMT file URLs
MSIGDB_GMT_URLS = {
    "all": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/msigdb.v2025.1.Hs.symbols.gmt",
    "hallmark": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt",
    "go": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/c5.go.v2025.1.Hs.symbols.gmt",
}


def parse_gmt(content: str, geneset_key: str = "geneset", genesymbols_key: str = "genesymbols") -> list[dict[str, Any]]:
    """
    Parse a GMT file content string into a list of dictionaries.

    Parameters
    ----------
        content : str
            Content of the GMT file as a string.
        geneset_key : str
            Key name for the gene set name in the output dictionaries.
        genesymbols_key : str
            Key name for the list of gene symbols in the output dictionaries.

    Returns
    -------
        list[dict[str, Any]]:
            List of dictionaries where each dictionary represents a gene set with its name and associated gene symbols.
    """
    records = []
    for line in StringIO(content):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        name, _, *genes = parts
        records.append({geneset_key: name, genesymbols_key: genes})
    return records


def fetch_gmt_content(url: str) -> str:
    """
    Download the content of a GMT file from a URL.

    Parameters
    ----------
        url : str
            URL to the GMT file.

    Returns
    -------
        str: Content of the GMT file as a string.
    """
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def read_gmt_file(filepath: str) -> str:
    """
    Read the content of a local GMT file.

    Parameters
    ----------
        filepath : str
            Path to the GMT file.

    Returns
    -------
        str: Content of the GMT file as a string.
    """
    with open(filepath, encoding="utf-8") as f:
        return f.read()


def get_gmt_dict(
    content: str, geneset_key: str = "geneset", genesymbols_key: str = "genesymbols"
) -> dict[str, list[str]]:
    """
    Parse GMT content and return a dictionary mapping gene set names to gene symbols.

    Parameters
    ----------
        content : str
            Content of the GMT file.
        geneset_key : str
            Key name for the gene set name in the output dictionary.
        genesymbols_key : str
            Key name for the list of gene symbols in the output dictionary.

    Returns
    -------
        dict[str, list[str]]:
            Dictionary where keys are gene set names and values are lists of gene symbols.
    """
    records = parse_gmt(content, geneset_key, genesymbols_key)
    return {rec[geneset_key]: rec[genesymbols_key] for rec in records}


def gmt_to_dataframe(content: str, geneset_key: str = "geneset", genesymbol_key: str = "genesymbol") -> pd.DataFrame:
    """
    Convert GMT content to a pandas DataFrame with one gene per row.

    Parameters
    ----------
        content : str
            Content of the GMT file.
        geneset_key : str
            Column name for the gene set name.
        genesymbol_key : str
            Column name for the gene symbol.

    Returns
    -------
        pd.DataFrame: DataFrame with columns [geneset_key, genesymbol_key], where each row corresponds to a gene in a gene set.
    """
    records = parse_gmt(content, geneset_key, "genesymbols")
    rows = [{geneset_key: rec[geneset_key], genesymbol_key: gene} for rec in records for gene in rec["genesymbols"]]
    return pd.DataFrame(rows)


def load_gmt(
    collection: str | None = None,
    url: str | None = None,
    filepath: str | None = None,
    geneset_key: str = "geneset",
    genesymbol_key: str = "genesymbol",
    min_genes: int = 5,
    max_genes: int = 500,
) -> pd.DataFrame:
    """
    Load a GMT file from a collection name, URL, or local file path and return as a DataFrame.

    Optionally filter gene sets by minimum and maximum gene counts.

    Parameters
    ----------
        collection : str | None
            Name of the MSigDB collection to load (e.g., 'hallmark', 'go', etc.).
            If None, must specify either url or filepath.
        url : str | None
            URL to a GMT file. If specified, collection must be None.
        filepath : str | None
            Path to a local GMT file. If specified, collection must be None.
        geneset_key : str
            Column name for the gene set name in the output DataFrame.
        genesymbol_key : str
            Column name for the gene symbol in the output DataFrame.
        min_genes : int
            Minimum number of genes in a gene set to include in the DataFrame.
        max_genes : int
            Maximum number of genes in a gene set to include in the DataFrame.

    Returns
    -------
        pd.DataFrame: DataFrame with columns [geneset_key, genesymbol_key].

    Raises
    ------
        ValueError: If none or more than one of collection, url, or filepath is specified.
    """
    sources = [collection, url, filepath]
    if sum(x is not None for x in sources) != 1:
        raise ValueError("Specify exactly one of collection, url, or filepath.")

    if collection is not None:
        if collection not in MSIGDB_GMT_URLS:
            raise ValueError(f"Unknown collection '{collection}'. Valid options: {list(MSIGDB_GMT_URLS.keys())}")
        content = fetch_gmt_content(MSIGDB_GMT_URLS[collection])
    elif url is not None:
        content = fetch_gmt_content(url)
    elif filepath is not None:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        content = read_gmt_file(filepath)
    else:
        raise ValueError("No source specified.")

    gmt = gmt_to_dataframe(content, geneset_key=geneset_key, genesymbol_key=genesymbol_key)
    geneset_sizes = gmt.groupby(geneset_key).size()
    valid_genesets = geneset_sizes.index[(geneset_sizes >= min_genes) & (geneset_sizes <= max_genes)]
    filtered_gmt = gmt[gmt[geneset_key].isin(valid_genesets)]
    return filtered_gmt
