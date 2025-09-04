import numpy as np
import pandas as pd


def get_de_genes(
    df: pd.DataFrame,
    group_key: str = "group",
    effect_key: str = "log2fc",
    pval_key: str = "pval",
    feature_key: str = "feature",
    effect_thresh: float = 1.0,
    pval_thresh: float = 0.01,
    top_n: int | None = None,
    return_labeled_df: bool = False,
) -> dict[str, dict[str, list]] | tuple[dict[str, dict[str, list]], pd.DataFrame]:
    """
    Analyze differential expression data: label significant genes and extract lists per group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing differential expression results.
    group_key : str, default="group"
        Column indicating the group or condition.
    effect_key : str, default="log2fc"
        Column with effect size values (e.g., log2 fold change, coefficient).
    pval_key : str, default="pval"
        Column with p-values.
    feature_key : str, default="feature"
        Column containing gene names.
    effect_thresh : float, default=1.0
        Threshold for absolute effect size.
    pval_thresh : float, default=0.01
        Threshold for significance.
    top_n : int or None, default=None
        Number of top genes to select per direction. If None, return all.
    return_labeled_df : bool, default=False
        If True, also return the labeled DataFrame.

    Returns
    -------
    dict or tuple
        If return_labeled_df is False:
            Dictionary of the form:
            {
                "group1": {"up": [...], "down": [...]},
                "group2": {"up": [...], "down": [...]},
                ...
            }
        If return_labeled_df is True:
            Tuple of (gene_lists_dict, labeled_dataframe)
    """
    # Create a copy to avoid modifying the original dataframe
    df_labeled = df.copy()

    # Add significance labels
    log_pval = -np.log10(df_labeled[pval_key])
    df_labeled["-log10(pval)"] = np.clip(log_pval, a_min=None, a_max=50)

    # Determine significance based on p-value and effect size thresholds
    sig_pval_mask = df_labeled[pval_key] < pval_thresh
    up_mask = sig_pval_mask & (df_labeled[effect_key] > effect_thresh)
    down_mask = sig_pval_mask & (df_labeled[effect_key] < -effect_thresh)

    df_labeled["significant"] = "NS"
    df_labeled.loc[up_mask, "significant"] = "Up"
    df_labeled.loc[down_mask, "significant"] = "Down"

    # Extract DE genes per group
    result = {}
    grouped = df_labeled.groupby(group_key)

    for group, sub_df in grouped:
        up_df = sub_df[sub_df["significant"] == "Up"]
        down_df = sub_df[sub_df["significant"] == "Down"]

        if top_n is not None:
            up_df = up_df.nlargest(top_n, effect_key)
            down_df = down_df.nsmallest(top_n, effect_key)

        result[group] = {
            "up": up_df[feature_key].tolist(),
            "down": down_df[feature_key].tolist(),
        }

    if return_labeled_df:
        return result, df_labeled
    else:
        return result
