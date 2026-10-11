import pandas as pd


def group_by_max(expr: pd.DataFrame) -> list[str]:
    """Order genes by group (column) of max expression."""
    max_group = expr.idxmax(axis=1)
    ordered = []
    for group in expr.columns:
        ordered.extend(max_group[max_group == group].index.tolist())
    return ordered
