import pandas as pd


def get_de_genes(
    df: pd.DataFrame,
    group_col: str = "group",
    coef_col: str = "coef",
    sig_col: str = "significant",
    gene_col: str = "feature",
    top_n: int | None = None,
) -> dict[str, dict[str, list]]:
    """
    Return up- and down-regulated genes per group, optionally limited to top-N by coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing differential expression results.
    group_col : str
        Column indicating the group or condition.
    coef_col : str
        Column with log fold change or coefficient values.
    sig_col : str
        Column indicating direction ("Up" or "Down").
    gene_col : str
        Column containing gene names.
    top_n : int or None
        Number of top genes to select per direction. If None, return all.

    Returns
    -------
    dict
        Dictionary of the form:
        {
            "group1": {"up": [...], "down": [...]},
            "group2": {"up": [...], "down": [...]},
            ...
        }
    """
    result = {}
    grouped = df.groupby(group_col)

    for group, sub_df in grouped:
        up_df = sub_df[sub_df[sig_col] == "Up"]
        down_df = sub_df[sub_df[sig_col] == "Down"]

        if top_n is not None:
            up_df = up_df.nlargest(top_n, coef_col)
            down_df = down_df.nsmallest(top_n, coef_col)

        result[group] = {
            "up": up_df[gene_col].tolist(),
            "down": down_df[gene_col].tolist(),
        }

    return result
