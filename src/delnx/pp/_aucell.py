import decoupler

from delnx.ds._gmt import load_gmt


def aucell(
    adata,
    collection: str | None = None,
    url: str | None = None,
    filepath: str | None = None,
    geneset_key: str = "geneset",
    genesymbol_key: str = "genesymbol",
    min_genes: int = 15,
    max_genes: int = 500,
):
    """
    Runs AUCell analysis on gene sets from a specified collection.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    collection : str, optional
        MSigDB collection name (e.g., 'hallmark', 'c2', 'c5', etc.).
    url : str, optional
        URL to download the gene set file.
    filepath : str, optional
        Path to a local gene set file.
    geneset_key : str, optional
        Column name for gene set names (default: "geneset").
    genesymbol_key : str, optional
        Column name for gene symbols (default: "genesymbol").
    min_genes : int, optional
        Minimum number of genes in a gene set to include (default: 15).
    max_genes : int, optional
        Maximum number of genes in a gene set to include (default: 500).

    Returns
    -------
    None
        The function modifies `adata` in place with AUCell results.
    """
    gmt = load_gmt(
        collection=collection,
        url=url,
        filepath=filepath,
        geneset_key=geneset_key,
        genesymbol_key=genesymbol_key,
        min_genes=min_genes,
        max_genes=max_genes,
    )
    gmt = gmt.rename(columns={geneset_key: "source", genesymbol_key: "target"})
    decoupler.mt.aucell(adata, gmt)
