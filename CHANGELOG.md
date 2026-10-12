# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `nb_fit()` / `nb_test()`: GPU-accelerated negative binomial GLM fitting with quasi-likelihood dispersion shrinkage (glmGamPoi approach).
- `nb_de()`: One-shot convenience wrapper around `nb_fit()` + `nb_test()`.
- `build_design()`: Shared design matrix builder with R-style formula support via patsy.
- `grouped()`: Per-group DE wrapper with cross-group multiple testing correction.
- `auroc()`: Pairwise AUROC effect size computation (now exported).
- Formula interface for `de()`, `nb_fit()`, and `nb_de()` via `formula=` parameter.
- `de()` now returns `log2fc` and `stat` columns alongside `coef`, `pval`, `padj`.
- Agent skill (`skills/delnx/`) and Claude Code plugin marketplace (`.claude-plugin/marketplace.json`) so AI coding agents can discover and use delnx; `llms.txt` served at the docs root.

### Changed

- `de()` no longer loops over pairwise comparisons. It builds a single design matrix and tests one contrast per call. Use `grouped()` for per-group analysis.
- `de()` signature simplified: removed `mode`, `min_samples`, `log2fc_threshold` parameters.
- ANOVA backend uses closed-form F-test via Frisch-Waugh-Lovell projection (no iterative fitting).
- AUROC in `_effects.py` uses Mann-Whitney U rank-sum statistic instead of sort+trapezoid.
- `rank_de()` p-values computed in JAX instead of SciPy.

### Changed (affects existing results)

- `de()` with the default `method="lr"` now fits a per-gene logistic regression with a likelihood ratio test, as documented. Before this release it silently ran the ANOVA F-test, so `lr` p-values change and `lr` runs are slower; results now depend on `maxiter`, and genes that perfectly separate the groups can get very large statistics.
- `de()` `coef`/`log2fc` (methods `lr`, `anova`, `anova_residual`) are now the regression coefficient of the contrast in expression ~ design (for log1p data, the log-scale mean difference); previous values were too small. `method="lr"` with a continuous contrast now raises instead of silently running ANOVA.

### Removed

- `de_enrichment_analysis()`: removed from public API.
- Pairwise comparison modes (`all_vs_ref`, `all_vs_all`, `1_vs_1`) from `de()`. These modes remain available in `log2fc()` and `auroc()`.
- cuML and PyDESeq2 backend references from documentation.

### Fixed

- `de()`: `method="lr"` and `method="anova_residual"` silently ran the ANOVA F-test; the requested test now runs (`"lr"` requires a binary contrast). `coef`/`log2fc` are now the OLS coefficient of the contrast in expression ~ design (previously the reversed-role regression of condition on expression, which underestimated fold changes).
- `compare()`: `"pairwise"` and `"all_vs_ref"` modes now pass an explicit `contrast` for the tested level. Previously `contrast` defaulted to the last design coefficient, which silently tested a covariate instead of the condition whenever `covariate_keys` was passed.
- `volcanoplot()` with `label_top > 0` no longer raises `KeyError: 'group'` on single-comparison results (e.g. plain `nb_test()` or `de()` output).
