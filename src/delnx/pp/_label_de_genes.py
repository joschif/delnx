import numpy as np
import pandas as pd


def label_de_genes(
    df: pd.DataFrame,
    coef_col: str = "coef",
    pval_col: str = "pval",
    auc_col: str = "auroc",
    coef_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    auc_thresh: float = 0,
) -> None:
    """
    Annotate a DE result dataframe with significance labels (in-place).

    Parameters
    ----------
    df : pd.DataFrame
        DE result dataframe with at least pval, coef, and auroc columns.
    coef_col : str, default="coef"
        Name of the coefficient column.
    pval_col : str, default="pval"
        Name of the p-value column.
    auc_col : str, default="auroc"
        Name of the AUROC column.
    coef_thresh : float, default=1.0
        Threshold for absolute log fold-change (effect size).
    pval_thresh : float, default=0.05
        Threshold for significance.
    auc_thresh : float, default=0
        Threshold for separation around AUROC=0.5.

    Returns
    -------
    None
        The original dataframe is modified in place to include:
        - '-log10(pval)'
        - 'separation'
        - 'significant': one of {'Up', 'Down', 'NS'}
    """
    log_pval = -np.log10(df[pval_col])
    df["-log10(pval)"] = np.clip(log_pval, a_min=None, a_max=50)
    df["separation"] = np.abs(df[auc_col] - 0.5)

    sig_mask = (df[pval_col] < pval_thresh) & (df["separation"] > auc_thresh)
    up_mask = sig_mask & (df[coef_col] > coef_thresh)
    down_mask = sig_mask & (df[coef_col] < -coef_thresh)

    df["significant"] = "NS"
    df.loc[up_mask, "significant"] = "Up"
    df.loc[down_mask, "significant"] = "Down"