from typing import Tuple, List
import pandas as pd


def get_top_de_genes(
    df: pd.DataFrame,
    top_n: int = 5,
    coef_col: str = "coef",
    sig_col: str = "significant",
) -> Tuple[List[str], List[str]]:
    """
    Select the top N up- and down-regulated genes by coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with DE results and a 'significant' label column.
    top_n : int, default=5
        Number of top genes to return for each direction.
    coef_col : str, default="coef"
        Column name for the coefficient.
    sig_col : str, default="significant"
        Column name for the significance label.

    Returns
    -------
    tuple of pd.DataFrame
        - Top upregulated genes
        - Top downregulated genes
    """
    top_up = df[df[sig_col] == "Up"].nlargest(top_n, coef_col)
    top_down = df[df[sig_col] == "Down"].nsmallest(top_n, coef_col)
    return top_up["feature"].tolist(), top_down["feature"].tolist()  # Ensure 'feature' column is present
