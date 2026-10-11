---
name: delnx
description: Differential expression (DE) analysis of single-cell and bulk genomics data in Python with delnx (GPU-accelerated via JAX). Use when testing genes between conditions or clusters in an AnnData object — pseudobulk negative binomial DE (glmGamPoi-style quasi-likelihood), linear-model / binomial DE on log-normalized or binary data, fast AUROC cluster markers, per-cell-type DE, formula designs with covariates, pseudobulking, size factors, log2 fold changes, gene set enrichment, and volcano / dot / violin plots.
---

# delnx

Python package for differential expression on AnnData (`import delnx as dx`). Docs: https://delnx.readthedocs.io

```bash
pip install delnx            # CPU
pip install "jax[cuda12]"    # optional: GPU acceleration (see JAX install docs)
```

Modules: `dx.pp` (preprocessing), `dx.tl` (tests and effect sizes), `dx.pl` (plots), `dx.ds` (gene sets, synthetic data).

## Pick the method

| Question | Data | Call |
|---|---|---|
| Condition effect with biological replicates (the default for "is gene X DE between treatment and control?") | Raw counts, pseudobulked | `dx.pp.pseudobulk` → `dx.tl.nb_fit` + `dx.tl.nb_test` (or one-shot `dx.tl.nb_de`) |
| Same, per cell type | Raw counts, pseudobulked by sample × cell type | `dx.tl.grouped(dx.tl.nb_de, pb, group_key="cell_type", ...)` |
| Cluster / cell-type markers (1-vs-all) | Non-negative: counts or log-normalized (not scaled / z-scored) | `dx.tl.rank_de(adata, condition_key="cell_type")` |
| Cell-level DE with covariates, log-normalized data | Log-normalized / scaled | `dx.tl.de(...)` (linear-model F-test; see note on effect sizes) |
| Binary data (e.g. ATAC peaks) | 0/1 | `dx.tl.de(method="binomial")` |
| Many comparisons (1-vs-all, pairwise, all-vs-reference) with joint FDR | Any `de`-compatible | `dx.tl.compare(dx.tl.de, adata, condition_key=..., mode=...)` |
| Effect sizes (also the way to get fold changes for `de` results) | Any | `dx.tl.log2fc`, `dx.tl.auroc` |
| Enrichment of a gene list | Gene symbols | `dx.tl.gsea(genes, background=...)` |

Rules that matter most:

1. **Replicates, not cells, are the unit for condition DE.** Testing conditions on single cells treats cells as independent and inflates significance. Pseudobulk by the biological replicate (`sample_key`) first, then use NB. Cell-level `de` is fine for markers and exploration, not for condition claims.
2. **NB needs raw integer counts.** `nb_fit`/`nb_de`/`pseudobulk` read `adata.X` unless `layer=` is given — pass `layer="counts"` when X is log-normalized. Never pass log-transformed data to NB.
3. **`de` expects log-normalized or scaled data**, not counts. Its p-values come from a linear-model F-test (in the current release `method="lr"`, `"anova"` and `"anova_residual"` all run this test whenever a design is built, which is always). Its `coef`/`log2fc` columns give the direction but **not a fold-change magnitude** — report fold changes from `dx.tl.log2fc(adata, condition_key=..., reference=..., mode="all_vs_ref")`.
4. **Fit once, test many**: with several contrasts on the same design, call `nb_fit` once and `nb_test` per contrast.
5. **Always pass `contrast` explicitly when the design has covariates** — `None` means the *last* coefficient, which is then a covariate, not your condition.

## Designs and contrasts

- `condition_key="treatment", reference="control"` builds `~ treatment` with treatment coding (`reference` becomes the intercept; default: the first category of a categorical column, otherwise the alphabetically first level).
- Covariates: `covariate_keys=["batch"]` with `condition_key`, or a formula: `formula="~ treatment + batch"` (patsy; `*` for interactions, continuous columns allowed). `condition_key` and `formula` are mutually exclusive.
- Coefficient names follow patsy: `["Intercept", "treatment[T.drugA]", "batch[T.b2]"]`. Inspect with `fit.design_column_names` or `dx.tl.build_design(adata.obs, formula=...)`.
- `contrast` accepts the full name `"treatment[T.drugA]"` (always works), the shorthand `"treatment[drugA]"`, the bare level `"drugA"` (when `condition_key` was used), an integer index, or `None` (last coefficient). Continuous covariates: `contrast="age"`. Interactions: `"treatment[T.drugA]:batch[T.b2]"`.
- **Level differences** (e.g. drugA vs drugB with control as reference): `nb_test` accepts difference contrasts and numeric vectors and runs a Wald test on the shared fit — `contrast="treatment[T.drugA] - treatment[T.drugB]"` or a numeric vector with one weight per entry of `fit.design_column_names`, in that order (it is not alphabetical — print it first). Use full bracketed names when level names contain `+` or `-`. `de` only takes a single coefficient: refit with `reference="drugB"` instead.

## Core recipes

```python
import delnx as dx

# Pseudobulk: one sample per biological replicate x cell type, summed raw counts
pb = dx.pp.pseudobulk(adata, sample_key="donor", group_key="cell_type", layer="counts", mode="sum")
# defaults drop samples with < 5 cells or < 5000 counts (min_cells, min_counts)

# NB DE for one cell type
sub = pb[pb.obs["cell_type"] == "T cell"].copy()
fit = dx.tl.nb_fit(sub, condition_key="treatment", reference="control", covariate_keys=["sex"])
res = dx.tl.nb_test(sub, fit, contrast="treatment[T.drugA]")

# NB DE for every cell type; padj re-corrected across groups
res = dx.tl.grouped(
    dx.tl.nb_de,
    pb,
    group_key="cell_type",
    condition_key="treatment",
    reference="control",
    contrast="treatment[T.drugA]",
)

# Markers
markers = dx.tl.rank_de(adata, condition_key="leiden")

# Cell-level DE with a covariate (log-normalized X); fold changes via log2fc
res = dx.tl.de(adata, formula="~ treatment + batch", contrast="treatment[T.drugA]")
lfc = dx.tl.log2fc(adata, condition_key="treatment", reference="control", mode="all_vs_ref")  # default mode is all_vs_all

# Plot
dx.pl.volcanoplot(res, pval_key="padj", pval_thresh=0.05, label_top=10)  # defaults use raw pval < 0.01
```

A tested end-to-end script on synthetic data is in `references/example_workflow.py`; full signatures and output columns are in `references/api.md`.

## Outputs

`nb_test`, `nb_de`, `de` return a DataFrame with `feature, log2fc, coef, stat, pval, padj` (`log2fc = coef / ln 2`, clipped to ±10; for NB `coef` is the natural-log fold change, for `de` see rule 3). `grouped` adds `group`; `compare` adds `comparison` (1_vs_all) or `test`/`reference` (pairwise, all_vs_ref) and re-computes `padj` across all comparisons. `rank_de` returns `feature, condition, auroc, z_score, pval, padj` (AUROC > 0.5 = higher in that group). Filter significance on `padj`, not `pval`.

## Performance

- Runs on CPU; much faster on GPU when JAX sees one (`import jax; jax.devices()`).
- `batch_size` (genes per batch) trades memory for speed — lower it on GPU out-of-memory errors.
- The first call compiles JAX code (seconds); later calls with similar shapes are fast. Don't benchmark the first call.
- NB on pseudobulk (tens–hundreds of samples) takes seconds to a minute for 20k genes; on 10k+ single cells it still works but prefer pseudobulk for condition DE anyway (rule 1).
- Joint NB model with many conditions: fit once, test each contrast. Keep the joint model for pseudobulk (it pools dispersion across all samples). Only for single-cell-sized inputs with ~80+ groups can separate 2-group fits be faster — and they test a different hypothesis (group vs rest, not vs reference).

## Common mistakes

- Contrast not found (`ValueError` listing the available columns): print `fit.design_column_names` and copy the exact name.
- Implausible NB results: check the input really is raw counts (`adata.X.max()`, integer values) and that `layer` points at them.
- `pseudobulk` returns almost nothing: `min_cells`/`min_counts` filtered the samples — lower them for small datasets.
- `compare` sets `condition_key`/`reference` itself (and `contrast` in `1_vs_all`). In `pairwise`/`all_vs_ref` mode it does not set `contrast`, so with covariates it tests the last covariate — don't combine those modes with covariates in the current release; loop over `de` calls with explicit `reference`/`contrast` instead.
- Comparing `padj` from separate `de` calls: use `grouped`/`compare` to correct jointly.
