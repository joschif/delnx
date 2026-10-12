"""End-to-end delnx workflow on synthetic data: pseudobulk -> NB DE per cell type, plus markers.

Runs in ~1 min on CPU. Doubles as a smoke test for the calls recommended in SKILL.md.

Usage: python example_workflow.py
"""

import delnx as dx

# Synthetic scRNA-seq: X is log-normalized, layers["counts"] holds raw counts.
# obs: cell_type, condition (control/treatment), condition_sample (= biological replicate)
adata = dx.ds.synthetic_adata(n_cells=3000, n_genes=500, n_samples=6)

# ---- 1. Condition DE: pseudobulk per replicate x cell type, then NB GLM ----
pb = dx.pp.pseudobulk(adata, sample_key="condition_sample", group_key="cell_type", layer="counts", mode="sum")
# pb.X now holds summed raw counts; obs keeps condition / cell_type
assert pb.n_obs > 0, "all pseudobulk samples were filtered (min_cells / min_counts)"

# One cell type: fit once, test the contrast
ct0 = pb[pb.obs["cell_type"] == "cell_type_0"].copy()
fit = dx.tl.nb_fit(ct0, condition_key="condition", reference="control", verbose=False)
print("design columns:", fit.design_column_names)  # ['Intercept', 'condition[T.treatment]']
res = dx.tl.nb_test(ct0, fit, contrast="condition[T.treatment]")
print(res.head())  # feature, log2fc, coef, stat, pval, padj

# All cell types at once; padj re-corrected across groups, adds a `group` column
res_all = dx.tl.grouped(
    dx.tl.nb_de,
    pb,
    group_key="cell_type",
    condition_key="condition",
    reference="control",
    contrast="condition[T.treatment]",
    verbose=False,
)
print(res_all.groupby("group")["padj"].apply(lambda p: (p < 0.05).sum()))

# ---- 2. Cluster markers on single cells (log-normalized X) ----
markers = dx.tl.rank_de(adata, condition_key="cell_type", verbose=False)
print(markers.sort_values("auroc", ascending=False).groupby("condition").head(3))

# ---- 3. Covariate-adjusted DE on log-normalized single cells (exploratory only: cells are not independent) ----
res_lr = dx.tl.de(adata, formula="~ condition + cell_type", contrast="condition[T.treatment]", verbose=False)
print(res_lr.head())  # default method="lr"; coef = log-scale mean difference, log2fc = coef / ln 2

# ---- 4. Volcano plot ----
dx.pl.volcanoplot(res, label_top=5, show=False)

assert {"feature", "log2fc", "pval", "padj"} <= set(res.columns)
assert "group" in res_all.columns and {"feature", "condition", "auroc", "pval"} <= set(markers.columns)
print("OK")
