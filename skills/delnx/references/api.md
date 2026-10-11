# delnx API reference

Generated from the delnx docstrings by `scripts/build_api_md.py` — do not edit by hand.

## `dx.pp`

### `dx.pp.aucell`

```python
aucell(adata, collection: str | None = None, url: str | None = None, filepath: str | None = None, geneset_key: str = 'geneset', genesymbol_key: str = 'genesymbol', min_genes: int = 15, max_genes: int = 500)
```

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

### `dx.pp.pseudobulk`

```python
pseudobulk(adata: anndata._core.anndata.AnnData, sample_key: str = 'batch', group_key: str | None = None, n_pseudoreps: int | None = None, layer: str | None = None, min_cells: int | None = 5, min_counts: int | None = 5000, mode: str = 'sum', **kwargs) -> anndata._core.anndata.AnnData
```

Create pseudobulk samples from single-cell RNA-seq data.

This function aggregates single-cell RNA-seq data into pseudobulk samples based on specified sample and group identifiers. It can optionally create random pseudoreplicates.

Parameters
----------
adata : :class:`~anndata.AnnData`
    AnnData object containing single-cell expression data.
sample_key : str, default="batch"
    Column name in `adata.obs` that contains the sample identifiers.
    Each unique value will become a separate pseudobulk sample
    (or multiple samples if `n_pseudoreps` is specified).
group_key : str | None, default=None
    Column name in `adata.obs` that contains group identifiers like cell types.
    If provided, pseudobulk samples will be generated separately for each combination
    of sample and group, enabling cell type-specific analysis.
n_pseudoreps : int | None, default=None
    Number of pseudoreplicates to create per sample. If :obj:`None`, uses the original
    sample structure without resampling. If specified, creates `n_pseudoreps`
    pseudoreplicates per sample by randomly sampling cells with replacement.
layer : str | None, default=None
    Layer in `adata.layers` to use for aggregation. If :obj:`None`, uses `adata.X`.
    Should contain raw or normalized counts, not log-transformed values.
mode : str, default="sum"
    Method to aggregate cell-level data into pseudobulk samples:
        - "sum": Sum of counts (recommended for RNA-seq data)
        - "mean": Average of counts across cells
min_cells : int | None, default=5
    Minimum number of cells required in a pseudobulk sample to retain it.
    Samples with fewer cells will be discarded.
min_counts : int | None, default=5000
    Minimum total counts required in a pseudobulk sample to retain it.
    Samples with fewer total counts will be discarded.
**kwargs
    Additional arguments to pass to :func:`decoupler.pp.pseudobulk`

Returns
-------
    :class:`~anndata.AnnData` object containing the pseudobulk data. The structure changes from
    cell-level to sample-level, with each row representing a pseudobulk sample.
    Original sample and group identifiers are preserved in the observations.

Notes
-----
- Wrapper around the decoupler (https://github.com/scverse/decoupler) pseudobulk function to support pseudo-replicates
- It's generally recommended to aggregate counts using the "sum" mode and then re-normalize rather than using "mean" directly on log-normalized data.

Examples
--------
Basic pseudobulking by sample:

>>> import scanpy as sc
>>> import delnx as dx
>>> adata = sc.read_h5ad("single_cell_data.h5ad")
>>> # Assuming adata.obs["sample"] contains sample identifiers
>>> pseudobulk_adata = dx.pp.pseudobulk(adata, sample_key="sample")

Pseudobulking by sample and cell type:

>>> # Create cell type-specific pseudobulk samples
>>> pseudobulk_adata = dx.pp.pseudobulk(adata, sample_key="sample", group_key="cell_type")

Creating pseudoreplicates for assessing technical variation:

>>> # Generate 5 pseudoreplicates per original sample
>>> pseudobulk_adata = dx.pp.pseudobulk(adata, sample_key="sample", n_pseudoreps=5)

### `dx.pp.size_factors`

```python
size_factors(adata, method='normed_sum', layer=None, obs_key_added='size_factors')
```

Compute size factors for (single-cell) RNA-seq normalization.

This function calculates sample/cell-specific normalization factors (size factors)
to account for differences in sequencing depth and technical biases between samples.
The computed size factors can be used to normalize counts for visualization or
as offset terms in statistical models for differential expression analysis.

Parameters
----------
adata : AnnData
    Annotated data matrix containing expression data.
method : str, default="normed_sum"
    Method to compute size factors:
    - "normed_sum": Library size normalization based on the total counts per
      sample. Simple and efficient, works with sparse matrices.
    - "ratio": DESeq2-style median-of-ratios size factors, robust to differential
      expression between samples. Requires dense matrices.
    - "poscounts": Positive counts method with geometric mean normalization.
      Requires dense matrices.
layer : str, optional
    Layer in `adata.layers` to use for size factor calculation. If None, uses
    `adata.X`. Should contain raw (unlogged) counts.
obs_key_added : str, default="size_factors"
    Key in `adata.obs` where the computed size factors will be stored.

Returns
-------
None
    Updates `adata` in place and sets the following field:
    - `adata.obs[obs_key_added]`: Size factors for each cell.

Examples
--------
Calculate library size normalization (default):

>>> import scanpy as sc
>>> import delnx as dx
>>> adata = sc.read_h5ad("counts.h5ad")
>>> dx.pp.size_factors(adata, method="normed_sum")

Calculate DESeq2-style median-of-ratios size factors:

>>> # Requires dense matrix
>>> if sparse.issparse(adata.X):
...     adata.X = adata.X.toarray()
>>> dx.pp.size_factors(adata, method="ratio", obs_key_added="ratio_factors")

Use size factors for normalization in differential expression analysis:

>>> # Compute DE with size factors as offset
>>> results = dx.tl.de(adata, condition_key="treatment", size_factor_key="size_factors")

Notes
-----
- Size factors are scaled to have a geometric mean of 1.0 across all samples
- Methods "ratio" and "poscounts" require dense matrices; use "normed_sum" for sparse data
- A warning will be raised if size factors cannot be computed for some cells
- Zero or invalid size factors are replaced with a small value (0.001)

## `dx.tl`

### `dx.tl.de`

```python
de(adata: anndata._core.anndata.AnnData, condition_key: str | None = None, formula: str | None = None, contrast: str | int | None = None, reference: str | None = None, covariate_keys: list[str] | None = None, method: str = 'lr', layer: str | None = None, multitest_method: str = 'fdr_bh', batch_size: int = 2048, maxiter: int = 100, verbose: bool = True) -> pandas.core.frame.DataFrame
```

Differential expression testing for non-count data.

General-purpose DE function for log-normalized, scaled, or binary data.
Builds a design matrix from ``formula`` or ``condition_key`` and tests
a single ``contrast``. For multiple comparisons, call this function
once per contrast or use :func:`grouped` for per-group analysis.

For count data, use :func:`nb_fit` + :func:`nb_test` instead.
For fast cluster markers, use :func:`rank_de`.

Parameters
----------
adata : AnnData
    Annotated data object containing expression data and metadata.
condition_key : str | None, default=None
    Column name in ``adata.obs`` containing condition labels.
    Internally builds ``"~ condition_key"`` for the design matrix.
    Mutually exclusive with ``formula``.
formula : str | None, default=None
    R-style formula for the design matrix (e.g., ``"~ treatment + batch"``).
    Parsed by patsy. Mutually exclusive with ``condition_key``.
contrast : str | int | None, default=None
    Coefficient to test. Supports shorthand:

    - Level name: ``"drugA"`` (resolved via ``condition_key``).
    - Bracket shorthand: ``"treatment[drugA]"`` (for multi-covariate formulas).
    - Full patsy name: ``"treatment[T.drugA]"`` (always works).
    - Integer index or None (last coefficient).
reference : str | None, default=None
    Reference level for categorical conditions. This level becomes
    the intercept in treatment coding.
covariate_keys : list[str] | None, default=None
    Columns in ``adata.obs`` to include as covariates.
    Only used with ``condition_key`` (include covariates in
    ``formula`` directly).
method : str, default="lr"
    Statistical method for testing:

        - ``"lr"``: Logistic regression predicting the (binary) contrast
          column from expression plus the remaining design columns, with a
          likelihood ratio test. Recommended for log-normalized single-cell
          data. Requires a binary contrast; use ``"anova"`` for continuous ones.
        - ``"anova"``: ANOVA based on linear model.
          Recommended for log-normalized or scaled data.
        - ``"anova_residual"``: Linear model with residual F-test.
        - ``"binomial"``: Binomial GLM likelihood ratio test.
          For binary data (e.g., ATAC-seq).
layer : str | None, default=None
    Layer in ``adata.layers`` containing expression data.
    If None, uses ``adata.X``.
multitest_method : str, default="fdr_bh"
    Method for multiple testing correction
    (see :func:`statsmodels.stats.multipletests`).
batch_size : int, default=2048
    Number of features to process per batch.
maxiter : int, default=100
    Maximum number of optimization iterations.
verbose : bool, default=True
    Whether to print progress messages.

Returns
-------
pd.DataFrame
    Results with columns:

    - ``feature``: Gene/feature names
    - ``log2fc``: ``coef / log(2)``, clipped to [-10, 10]. A log2 fold change
      for natural-log (e.g. log1p) data; for ``"binomial"`` a log2 odds ratio.
    - ``coef``: Effect size of the contrast. For ``"lr"``, ``"anova"`` and
      ``"anova_residual"``: OLS coefficient of the contrast column when
      regressing expression on the full design (difference in mean
      expression, per unit for continuous contrasts). For ``"binomial"``:
      log odds ratio from the binomial GLM.
    - ``stat``: Test statistic of the selected method (LR chi-squared for
      ``"lr"``, partial F for ``"anova"``, residual-variance F ratio for
      ``"anova_residual"``, deviance reduction vs. the intercept-only model
      for ``"binomial"``, whose ``pval`` is a Wald test)
    - ``pval``: Raw p-value
    - ``padj``: Adjusted p-value

Examples
--------
Simple condition comparison:

>>> results = dx.tl.de(adata, condition_key="treatment", reference="control", contrast="treatment[T.drugA]")

Formula-based with covariates:

>>> results = dx.tl.de(adata, formula="~ treatment + batch", contrast="treatment[T.drugA]")

Continuous covariate:

>>> results = dx.tl.de(adata, formula="~ age + sex", contrast="age", method="anova")

Binomial for binary ATAC data:

>>> results = dx.tl.de(
...     adata,
...     condition_key="treatment",
...     reference="control",
...     contrast="treatment[T.drugA]",
...     method="binomial",
...     layer="binary",
... )

Notes
-----
For count data (RNA-seq), use :func:`nb_fit` + :func:`nb_test` instead.
For fast cluster markers, use :func:`rank_de`.

### `dx.tl.grouped`

```python
grouped(func: collections.abc.Callable[..., pandas.core.frame.DataFrame], adata: anndata._core.anndata.AnnData, group_key: str, min_samples: int = 2, multitest_method: str = 'fdr_bh', verbose: bool = True, **kwargs) -> pandas.core.frame.DataFrame
```

Run a DE function separately for each group and combine results.

Thin orchestrator that subsets ``adata`` by each unique value of
``adata.obs[group_key]``, calls ``func`` on each subset, and
re-corrects p-values across all groups.

Parameters
----------
func : callable
    DE function with signature ``func(adata, **kwargs) -> pd.DataFrame``.
    The returned DataFrame must contain a ``pval`` column.
    Works with :func:`de`, :func:`rank_de`, or any custom function
    (e.g., a lambda wrapping :func:`nb_fit` + :func:`nb_test`).
adata : AnnData
    Annotated data object.
group_key : str
    Column in ``adata.obs`` defining groups (e.g., ``"cell_type"``).
min_samples : int, default=2
    Minimum observations per group to run the analysis. Groups with
    fewer observations are skipped with a warning.
multitest_method : str, default="fdr_bh"
    Method for multiple testing correction across all groups
    (see :func:`statsmodels.stats.multipletests`).
verbose : bool, default=True
    Whether to print progress messages.
**kwargs
    Passed through to ``func``.

Returns
-------
pd.DataFrame
    Combined results with an additional ``group`` column. The ``padj``
    column is re-computed across all groups.

Examples
--------
Per-cell-type logistic regression:

>>> results = dx.tl.grouped(dx.tl.de, adata, group_key="cell_type",
...                         condition_key="treatment", reference="control",
...                         contrast="treatment[T.drugA]")

Per-cell-type rank-based markers:

>>> results = dx.tl.grouped(dx.tl.rank_de, adata, group_key="cell_type", condition_key="treatment")

Per-cell-type negative binomial DE:

>>> def nb_de(adata, **kw):
...     fit = dx.tl.nb_fit(adata, **kw)
...     return dx.tl.nb_test(adata, fit)
>>> results = dx.tl.grouped(nb_de, adata, group_key="cell_type", condition_key="treatment")

### `dx.tl.build_design`

```python
build_design(obs: pandas.core.frame.DataFrame, formula: str | None = None, condition_key: str | None = None, reference: str | None = None, covariate_keys: list[str] | None = None) -> tuple[numpy.ndarray, list[str]]
```

Build a design matrix from a formula or condition_key.

Parameters
----------
obs : pd.DataFrame
    Observation metadata (typically ``adata.obs``).
formula : str | None
    R-style formula (e.g., ``"~ treatment + batch"``).
    Mutually exclusive with ``condition_key``.
condition_key : str | None
    Column in ``obs`` for condition labels. Builds a formula internally.
    Mutually exclusive with ``formula``.
reference : str | None
    Reference level for the condition. Used to reorder categorical levels
    so the reference becomes the intercept. Only used with ``condition_key``
    or when ``formula`` contains a simple term matching a categorical column.
covariate_keys : list[str] | None
    Additional columns to include as covariates. Only used with
    ``condition_key`` (ignored when ``formula`` is provided, since
    covariates should be part of the formula).

Returns
-------
tuple[np.ndarray, list[str]]
    Design matrix of shape ``(n_obs, n_coef)`` and list of column names.

Raises
------
ValueError
    If both ``formula`` and ``condition_key`` are specified, or neither is.

### `dx.tl.compare`

```python
compare(func: collections.abc.Callable[..., pandas.core.frame.DataFrame], adata: anndata._core.anndata.AnnData, condition_key: str, mode: str = '1_vs_all', reference: str | None = None, min_samples: int = 2, multitest_method: str = 'fdr_bh', verbose: bool = True, **kwargs) -> pandas.core.frame.DataFrame
```

Run DE comparisons across multiple groups.

Orchestrator that constructs 1-vs-all, pairwise, or all-vs-reference
comparisons, calls ``func`` for each, and re-corrects p-values across
all comparisons.

Parameters
----------
func : callable
    DE function with signature ``func(adata, **kwargs) -> pd.DataFrame``.
    The returned DataFrame must contain a ``pval`` column.
    Works with :func:`de`, :func:`nb_de`, or any custom function.
adata : AnnData
    Annotated data object.
condition_key : str
    Column in ``adata.obs`` defining groups (e.g., ``"cell_type"``).
mode : str, default="1_vs_all"
    Comparison strategy:

    - ``"1_vs_all"``: Test each group against all others (marker detection).
    - ``"pairwise"``: Test all pairs of groups.
    - ``"all_vs_ref"``: Test each group against a single reference.
reference : str | None, default=None
    Reference level. Required for ``"all_vs_ref"`` mode.
min_samples : int, default=2
    Minimum observations per group/subset. Comparisons with fewer
    observations are skipped.
multitest_method : str, default="fdr_bh"
    Method for multiple testing correction across all comparisons
    (see :func:`statsmodels.stats.multipletests`).
verbose : bool, default=True
    Whether to print progress messages.
**kwargs
    Passed through to ``func``. Do not pass ``condition_key``,
    ``reference``, or ``contrast`` — these are set by ``compare()``.

Returns
-------
pd.DataFrame
    Combined results with additional comparison columns. The ``padj``
    column is re-computed across all comparisons.

Examples
--------
1-vs-all marker detection with logistic regression:

>>> markers = dx.tl.compare(dx.tl.de, adata, condition_key="cell_type",
...                         mode="1_vs_all", method="lr")

All pairwise comparisons:

>>> pairwise = dx.tl.compare(dx.tl.de, adata, condition_key="cell_type",
...                          mode="pairwise", method="anova")

All groups vs control:

>>> vs_ctrl = dx.tl.compare(dx.tl.de, adata, condition_key="treatment",
...                         mode="all_vs_ref", reference="control")

### `dx.tl.resolve_contrast`

```python
resolve_contrast(contrast: str | int | None, column_names: list[str], condition_key: str | None = None) -> int
```

Resolve a contrast specification to a column index.

Supports several shorthand forms so users don't need to write full
patsy treatment-coding names:

1. Full patsy name: ``"treatment[T.drugA]"`` — exact match.
2. Bracket shorthand: ``"treatment[drugA]"`` — resolves to
   ``"treatment[T.drugA]"``.
3. Bare level with ``condition_key``: ``"drugA"`` — resolves to
   ``"treatment[T.drugA]"`` when ``condition_key="treatment"``.
4. Bare level scan: ``"drugA"`` — if exactly one column ends with
   ``[T.drugA]``, use it.

Parameters
----------
contrast : str | int | None
    Contrast to resolve. ``None`` uses the last coefficient.
column_names : list[str]
    Design matrix column names (from :func:`build_design`).
condition_key : str | None
    Condition key, used for bare-level resolution.

Returns
-------
int
    Column index into the design matrix.

### `dx.tl.log2fc`

```python
log2fc(adata: anndata._core.anndata.AnnData, condition_key: str, reference: str | tuple[str, str] | None = None, mode: Literal['all_vs_ref', 'all_vs_all', '1_vs_1', 'continuous'] = 'all_vs_all', layer: str | None = None, data_type: Literal['counts', 'lognorm', 'binary', 'scaled', 'auto'] = 'auto', min_samples: int = 2, verbose: bool = True) -> pandas.core.frame.DataFrame
```

Calculate log2 fold changes between condition levels.

This function computes log2 fold changes (log2FC) between different experimental
conditions for all features in the dataset. It supports various comparison modes,
different data types, and can normalize by size factors when appropriate.

Parameters
----------
adata : AnnData
    AnnData object containing expression data and metadata.
condition_key : str
    Column name in `adata.obs` containing condition labels.
reference : str | tuple[str, str] | None, default=None
    Reference condition for comparisons, specified as:
    - Single string: reference condition for all comparisons
    - Tuple (reference, comparison): specific pair to compare
    - None: automatically determined based on mode parameter
mode : ComparisonMode, default="all_vs_all"
    Comparison strategy:
    - "all_vs_ref": Compare all condition levels against reference level
    - "all_vs_all": Compare all pairs of condition levels
    - "1_vs_1": Compare only reference vs comparison (requires tuple reference)
layer : str | None, default=None
    Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
data_type : DataType, default="auto"
    Type of expression data:
    - "auto": Automatically infer from data characteristics
    - "counts": Raw count data
    - "lognorm": Log-normalized data (log1p of normalized counts)
    - "binary": Binary expression data (0/1)
min_samples : int, default=2
    Minimum number of samples required per condition level.
    Comparisons with fewer samples are skipped.
verbose : bool, default=True
    Whether to print progress information and data type inference results.

Returns
-------
pd.DataFrame
    DataFrame containing log2 fold change results with columns:
    - "feature": Feature/gene names
    - "test_condition": Test condition label
    - "ref_condition": Reference condition label
    - "log2fc": Log2 fold change values (positive means up-regulated in test condition)

Examples
--------
Basic usage with automatic data type inference:

>>> import scanpy as sc
>>> import delnx as dx
>>> adata = sc.read_h5ad("dataset.h5ad")
>>> results = dx.tl.log2fc(adata, condition_key="treatment")

Comparing specific conditions:

>>> results = dx.tl.log2fc(adata, condition_key="treatment", reference=("control", "treated"), mode="1_vs_1")

### `dx.tl.auroc`

```python
auroc(adata: anndata._core.anndata.AnnData, condition_key: str, reference: str | tuple[str, str] | None = None, mode: Literal['all_vs_ref', 'all_vs_all', '1_vs_1', 'continuous'] = 'all_vs_all', layer: str | None = None, min_samples: int = 2, batch_size: int = 2048, verbose: bool = False) -> pandas.core.frame.DataFrame
```

Calculate Area Under the Receiver Operating Characteristic (AUROC) between condition levels.

This function computes AUROC values for all features between different experimental
conditions. AUROC quantifies how well a feature's expression can distinguish between
two conditions, providing a measure of the feature's discriminative power independent
of any specific threshold.

Parameters
----------
adata : AnnData
    AnnData object containing expression data and metadata.
condition_key : str
    Column name in `adata.obs` containing condition labels.
reference : str | tuple[str, str] | None, default=None
    Reference condition for comparisons, specified as:
    - Single string: reference condition for all comparisons
    - Tuple (reference, comparison): specific pair to compare
    - None: automatically determined based on mode parameter
mode : ComparisonMode, default="all_vs_all"
    Comparison strategy:
    - "all_vs_ref": Compare all condition levels against reference level
    - "all_vs_all": Compare all pairs of condition levels
    - "1_vs_1": Compare only reference vs comparison (requires tuple reference)
layer : str | None, default=None
    Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
min_samples : int, default=2
    Minimum number of samples required per condition level.
    Comparisons with fewer samples are skipped.
batch_size : int, default=2048
    Number of features to process per batch. Adjust based on available memory
    and dataset size.
verbose : bool, default=False
    Whether to print progress information.

Returns
-------
pd.DataFrame
    DataFrame containing AUROC results with columns:
    - "feature": Feature/gene names
    - "test_condition": Test condition label
    - "ref_condition": Reference condition label
    - "auroc": AUROC values (0.5=random, 1=perfect separation with higher values in test)

Examples
--------
Basic usage for all pairwise comparisons:

>>> import scanpy as sc
>>> import delnx as dx
>>> adata = sc.read_h5ad("dataset.h5ad")
>>> results = dx.tl.auroc(adata, condition_key="cell_type")

Looking at specific condition comparisons:

>>> # Compare only CD4+ T cells vs CD8+ T cells
>>> results = dx.tl.auroc(adata, condition_key="cell_type", reference=("CD4+ T", "CD8+ T"), mode="1_vs_1")

>>> # Compare all cell types against a reference type
>>> results = dx.tl.auroc(adata, condition_key="cell_type", reference="B cells", mode="all_vs_ref")

Notes
-----
- AUROC values range from 0 to 1, where:
  - 0.5 indicates the feature cannot distinguish between conditions (random)
  - Values >0.5 indicate higher expression in the test condition
  - Values <0.5 indicate higher expression in the reference condition
- The implementation uses JAX for accelerated computation and batch processing
  to efficiently handle large datasets

### `dx.tl.gsea`

```python
gsea(genes: collections.abc.Sequence[str], background: collections.abc.Sequence[str] | None = None, gene_sets: dict[str, list[str]] | None = None, collection: str = 'all', url: str | None = None, filepath: str | None = None, geneset_key: str = 'geneset', genesymbol_key: str = 'genesymbol', method: str = 'enrichr', return_object: bool = False, min_genes: int = 5, max_genes: int = 500) -> pandas.core.frame.DataFrame | typing.Any
```

Run enrichment analysis for a single gene list using Enrichr.

Parameters
----------
genes : Sequence[str]
    List of gene symbols to analyze.
background : Sequence[str], optional
    Background gene list to use for enrichment analysis. If None, uses all genes in the gene sets.
gene_sets : dict[str, list[str]], optional
    Pre-loaded gene sets as a dictionary where keys are gene set names and values are lists of gene symbols.
    If None, will load gene sets based on the provided collection, URL, or filepath.
collection : str
    Name of the collection to load gene sets from. Default is "all".
url : str, optional
    URL to load the GMT file from. If None, uses the default collection.
filepath : str, optional
    Local file path to load the GMT file from. If None, uses the default collection.
geneset_key : str
    Column name for the gene set name in the output dictionary. Default is "geneset".
genesymbol_key : str
    Column name for the gene symbol in the output dictionary. Default is "genesymbol".
method : str
    Method to use for enrichment analysis. Currently only "enrichr" is supported.
return_object : bool
    If True, returns the gseapy Enrichr object. If False, returns a pandas DataFrame with results.
min_genes : int
    Minimum number of genes in a gene set to include in the analysis. Default is 5.
max_genes : int
    Maximum number of genes in a gene set to include in the analysis. Default is 500.

### `dx.tl.rank_de`

```python
rank_de(adata: anndata._core.anndata.AnnData, condition_key: str, layer: str | None = None, use_ties: bool = False, multitest_method: str = 'fdr_bh', n_cpus: int | None = None, min_samples: int = 2, batch_size: int | None = 512, verbose: bool = True) -> pandas.core.frame.DataFrame
```

Perform rank-based differential expression analysis using AUROC statistics.

This function identifies differentially expressed features between conditions using
Area Under the ROC Curve (AUROC) analysis on ranked gene expression data. It
efficiently handles sparse matrices with memory-optimized batch processing and
provides optional tie correction for improved statistical accuracy.

Parameters
----------
adata : AnnData
    AnnData object containing expression data and metadata.
condition_key : str
    Column name in `adata.obs` containing condition labels.
layer : str | None, default=None
    Layer in `adata.layers` to use for expression data. If None, uses `adata.X`.
use_ties : bool, default=False
    Whether to apply tie correction to p-value calculations. Recommended for
    sparse data with many identical values (especially zeros).
multitest_method : str, default='fdr_bh'
    Method for multiple testing correction. Accepts any method supported by :func:`statsmodels.stats.multipletests`. Common options include:
        - "fdr_bh": Benjamini-Hochberg FDR correction
        - "bonferroni": Bonferroni correction
n_cpus : int | None, default=None
    Number of CPU cores for parallel processing. If None, uses available threads.
    Parallel processing is enabled when n_cpus >= 4 or use_ties=True.
min_samples : int, default=2
    Minimum samples required per condition. Conditions with fewer samples excluded.
batch_size : int | None, default=512
    Features per batch. Larger values use more memory but may be more efficient. If None, processes all features at once.
verbose : bool, default=True
    Whether to show progress and algorithm information.

Returns
-------
pd.DataFrame
    Results with columns:
    - "feature": Feature/gene names
    - "condition": Condition label (one-vs-all comparison)
    - "auroc": AUROC values (0.5=random, >0.5=upregulated, <0.5=downregulated)
    - "z_score": Absolute Mann-Whitney U z-statistic (direction from "auroc"; tie-corrected if ``use_ties=True``)
    - "pval": Two-tailed p-values from Mann-Whitney U test
    - "padj": Adjusted p-values (``multitest_method``)

Examples
--------
Basic one-vs-all differential expression:

>>> import delnx as dx
>>> results = dx.tl.rank_de(adata, condition_key="cell_type")

With tie correction for improved p-values:

>>> results = dx.tl.rank_de(adata, condition_key="treatment", use_ties=True)

Custom batch size for memory optimization:

>>> results = dx.tl.rank_de(adata, condition_key="condition", batch_size=1024)

Notes
-----
This implementation uses several optimizations:
- Numba JIT compilation for fast ranking algorithms
- JAX acceleration for AUROC calculations
- Memory-efficient batch processing to handle large datasets
- Automatic algorithm selection based on data characteristics
- Optional tie correction for statistical accuracy with sparse data

### `dx.tl.nb_fit`

```python
nb_fit(adata: anndata._core.anndata.AnnData, condition_key: str | None = None, formula: str | None = None, design: numpy.ndarray | None = None, design_column_names: list[str] | None = None, reference: str | None = None, covariate_keys: list[str] | None = None, size_factors: str | numpy.ndarray | None = 'normed_sum', layer: str | None = None, overdispersion: bool = True, batch_size: int = 512, maxiter: int = 100, verbose: bool = True, overdispersion_shrinkage: bool = True, do_cox_reid_adjustment: bool = True) -> delnx.tl._glm_gp.NBFitResult
```

Fit Gamma-Poisson (negative binomial) GLMs to count data.

This implements the glmGamPoi approach for fast and accurate differential
expression analysis using GPU-accelerated Newton-Raphson with quasi-likelihood
dispersion shrinkage.

Parameters
----------
adata : AnnData
    AnnData object containing count data.
condition_key : str | None, default=None
    Column in ``adata.obs`` for condition labels. Creates a design matrix
    with intercept + condition indicators. Mutually exclusive with ``formula``.
formula : str | None, default=None
    R-style formula for the design matrix (e.g., ``"~ treatment + batch"``
    or ``"~ treatment * batch"``). Parsed by patsy. Mutually exclusive
    with ``condition_key``.
design : np.ndarray | None, default=None
    Custom design matrix. If provided, overrides ``condition_key``,
    ``formula``, ``reference``, and ``covariate_keys``.
    Shape should be (n_samples, n_coefficients).
design_column_names : list[str] | None, default=None
    Column names for a custom ``design`` matrix. Enables string-based
    ``contrast`` in :func:`nb_test`. If None with custom ``design``,
    generic names ``coef_0``, ``coef_1``, ... are used.
reference : str | None, default=None
    Reference level for the condition. This level becomes the intercept.
    If None, the alphabetically first level is used.
covariate_keys : list[str] | None, default=None
    Columns in ``adata.obs`` to include as covariates in the design matrix.
    Only used with ``condition_key`` (include covariates in ``formula`` directly).
size_factors : str | np.ndarray | None, default="normed_sum"
    Size factors for normalization. Can be:

    - ``"normed_sum"``: Compute using normalized sum method
    - ``"poscounts"``: DESeq2-style positive counts method
    - np.ndarray: Pre-computed size factors
    - None: No size factor normalization (all 1s)
layer : str | None, default=None
    Layer in ``adata.layers`` containing counts. If None, uses ``adata.X``.
overdispersion : bool, default=True
    Whether to estimate overdispersion. If False, uses Poisson (disp=0).
batch_size : int, default=512
    Number of genes to process in each batch.
maxiter : int, default=100
    Maximum iterations for Newton-Raphson.
verbose : bool, default=True
    Whether to show progress.
overdispersion_shrinkage : bool, default=True
    Whether to apply quasi-likelihood shrinkage to dispersions.
do_cox_reid_adjustment : bool, default=True
    Whether to apply Cox-Reid adjustment to dispersion MLE.

Returns
-------
NBFitResult
    Fitted model results containing coefficients, dispersions, and
    fitted values.

Examples
--------
Basic usage with condition comparison:

>>> import delnx as dx
>>> fit = dx.tl.nb_fit(adata, condition_key="treatment")
>>> results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugA]")

With reference level and covariates:

>>> fit = dx.tl.nb_fit(adata, condition_key="treatment", reference="control", covariate_keys=["batch", "sex"])

Formula-based design with interactions:

>>> fit = dx.tl.nb_fit(adata, formula="~ treatment * batch")
>>> results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugA]:batch[T.batch2]")

Continuous covariate:

>>> fit = dx.tl.nb_fit(adata, formula="~ age + sex")
>>> results = dx.tl.nb_test(adata, fit, contrast="age")

### `dx.tl.nb_test`

```python
nb_test(adata: anndata._core.anndata.AnnData, fit: delnx.tl._glm_gp.NBFitResult, contrast: str | int | None = None, reduced_design: numpy.ndarray | None = None, multitest_method: str = 'fdr_bh', lfc_threshold: float = 0.0, batch_size: int = 512) -> pandas.core.frame.DataFrame
```

Test for differential expression using quasi-likelihood F-test.

Parameters
----------
adata : AnnData
    AnnData object (same one passed to ``nb_fit()``).
fit : NBFitResult
    Fitted model from ``nb_fit()``.
contrast : str | int | None, default=None
    Coefficient to test. Supports shorthand:

    - Level name: ``"drugA"`` (resolved via ``condition_key``).
    - Bracket shorthand: ``"treatment[drugA]"`` (for multi-covariate models).
    - Full patsy name: ``"treatment[T.drugA]"`` (always works).
    - Integer index or None (last coefficient).
    - None: Test last coefficient.
reduced_design : np.ndarray | None, default=None
    Reduced design matrix for likelihood ratio test.
    If None, automatically created by dropping the tested coefficient.
multitest_method : str, default="fdr_bh"
    Method for multiple testing correction.
lfc_threshold : float, default=0.0
    Threshold for log2 fold change filtering.
batch_size : int, default=512
    Number of genes to process in each batch. Controls GPU memory usage
    for the reduced model refit.

Returns
-------
pd.DataFrame
    Results with columns:

    - ``feature``: Gene/feature names
    - ``log2fc``: Log2 fold change
    - ``coef``: Model coefficient
    - ``stat``: F-statistic
    - ``pval``: Raw p-value
    - ``padj``: Adjusted p-value

Examples
--------
Test for treatment effect:

>>> fit = dx.tl.nb_fit(adata, condition_key="treatment")
>>> results = dx.tl.nb_test(adata, fit)

Test specific contrast by name:

>>> results = dx.tl.nb_test(adata, fit, contrast="treatment[T.drugB]")

### `dx.tl.nb_de`

```python
nb_de(adata: anndata._core.anndata.AnnData, condition_key: str | None = None, formula: str | None = None, design: numpy.ndarray | None = None, design_column_names: list[str] | None = None, reference: str | None = None, covariate_keys: list[str] | None = None, size_factors: str | numpy.ndarray | None = 'normed_sum', layer: str | None = None, contrast: str | int | None = None, multitest_method: str = 'fdr_bh', lfc_threshold: float = 0.0, overdispersion: bool = True, batch_size: int = 512, maxiter: int = 100, verbose: bool = True, overdispersion_shrinkage: bool = True, do_cox_reid_adjustment: bool = True) -> pandas.core.frame.DataFrame
```

One-shot negative binomial DE: fit model and test in one call.

Convenience wrapper around :func:`nb_fit` + :func:`nb_test`.
For reusing a fit across multiple contrasts, call them separately.

Parameters
----------
adata : AnnData
    AnnData object containing count data.
condition_key : str | None, default=None
    Column in ``adata.obs`` for condition labels. Mutually exclusive
    with ``formula``.
formula : str | None, default=None
    R-style formula for the design matrix (e.g., ``"~ treatment + batch"``).
    Mutually exclusive with ``condition_key``.
design : np.ndarray | None, default=None
    Custom design matrix. Overrides ``condition_key`` and ``formula``.
design_column_names : list[str] | None, default=None
    Column names for a custom ``design`` matrix.
reference : str | None, default=None
    Reference level for the condition.
covariate_keys : list[str] | None, default=None
    Columns in ``adata.obs`` to include as covariates.
size_factors : str | np.ndarray | None, default="normed_sum"
    Size factors for normalization.
layer : str | None, default=None
    Layer in ``adata.layers`` containing counts.
contrast : str | int | None, default=None
    Contrast to test (passed to :func:`nb_test`).
multitest_method : str, default="fdr_bh"
    Method for multiple testing correction.
lfc_threshold : float, default=0.0
    Minimum absolute log2 fold change threshold.
overdispersion : bool, default=True
    Whether to estimate overdispersion.
batch_size : int, default=512
    Number of genes per batch.
maxiter : int, default=100
    Maximum iterations for Newton-Raphson.
verbose : bool, default=True
    Whether to show progress.
overdispersion_shrinkage : bool, default=True
    Whether to apply quasi-likelihood shrinkage.
do_cox_reid_adjustment : bool, default=True
    Whether to apply Cox-Reid adjustment.

Returns
-------
pd.DataFrame
    DE results (same as :func:`nb_test`).

Examples
--------
Simple condition comparison:

>>> results = dx.tl.nb_de(adata, condition_key="treatment", reference="control")

Formula-based with covariates:

>>> results = dx.tl.nb_de(adata, formula="~ treatment + batch", contrast="treatment[T.drugA]")

## `dx.pl`

### `dx.pl.dotplot`

```python
dotplot(adata: Any, markers: collections.abc.Sequence[str], groupby: str | list[str], save: str | None = None, **kwargs)
```

Create a dot plot showing mean expression and fraction of cells expressing markers per group.

Parameters
----------
adata : AnnData
    Annotated data matrix.
markers : sequence of str
    Marker genes/features to plot.
groupby: Key(s) in adata.obs to group by.
**kwargs
    Additional arguments passed to DotPlot.

### `dx.pl.heatmap`

```python
heatmap(adata: Any, markers: collections.abc.Sequence[str], groupby: str | list[str], save: str | None = None, **kwargs)
```

Create a dot plot showing mean expression and fraction of cells expressing markers per group.

Parameters
----------
adata : AnnData
    Annotated data matrix.
markers : sequence of str
    Marker genes/features to plot.
groupby: Key(s) in adata.obs to group by.
save : str, optional
    If provided, saves the plot to the specified file path.
**kwargs
    Additional arguments passed to DotPlot.

Returns
-------
    None

### `dx.pl.matrixplot`

```python
matrixplot(adata: Any, markers: collections.abc.Sequence[str], groupby: str | list[str], save: str | None = None, **kwargs)
```

Create a matrix plot showing mean expression of markers per group.

Parameters
----------
adata : AnnData
    Annotated data matrix.
markers : sequence of str
    Marker genes/features to plot.
groupby: Key(s) in adata.obs to group by.
**kwargs
    Additional arguments passed to DotPlot.

### `dx.pl.violinplot`

```python
violinplot(adata: anndata._core.anndata.AnnData, genes: list[str], groupby: str, splitby: str | None = None, plot_type: Literal['violin', 'box', 'boxen'] = 'violin', save: str | None = None, show: bool = True, **kwargs) -> marsilea.base.StackBoard
```

Create a violin plot for gene expression across groups and splits.

Parameters
----------
adata : AnnData
    Annotated data matrix.
genes : list[str]
    List of gene names to plot.
groupby : str
    Column name in `adata.obs` to group by.
splitby : str | None, optional
    Column name in `adata.obs` to split by.
plot_type : Literal["violin", "box", "boxen"], default "violin"
    Type of plot to generate.
save : str | None, optional
    Path to save the figure. If None, does not save.
show : bool, default True
    Whether to display the plot.
**kwargs
    Additional arguments passed to ViolinPlot.

### `dx.pl.volcanoplot`

```python
volcanoplot(df: pandas.core.frame.DataFrame, x: str = 'log2fc', y: str = '-log10(pval)', effect_key: str = 'log2fc', pval_key: str = 'pval', feature_key: str = 'feature', effect_thresh: float = 0.5, pval_thresh: float = 0.01, thresh: dict[str, float] | None = None, label_top: int = 0, color_legend_title: str | None = None, ax: matplotlib.axes._axes.Axes | None = None, figsize: tuple[float, float] = (8, 6), show: bool | None = True, save: str | bool | None = None, return_fig: bool = False)
```

Create a volcano plot using matplotlib. Labels significant genes based on thresholds or uses "significant" column if present.

Parameters
----------
df : pd.DataFrame
    DataFrame containing differential expression results.
x : str, default="log2fc"
    Column name for x-axis (effect size).
y : str, default="-log10(pval)"
    Column name for y-axis (significance).
effect_key : str, default="log2fc"
    Column with effect size values for DE analysis.
pval_key : str, default="pval"
    Column with p-values for DE analysis.
feature_key : str, default="feature"
    Column containing gene names for labeling.
effect_thresh : float, default=0.5
    Threshold for absolute effect size.
pval_thresh : float, default=0.01
    Threshold for significance.
thresh : dict[str, float] or None
    Dictionary mapping axis names to threshold values for plot lines,
    e.g. {'log2fc': 1.0, '-log10(pval)': 1.3}.
    If None, uses effect_thresh and -log10(pval_thresh).
label_top : int, default=0
    If > 0, label top N up/down genes by effect size.
color_legend_title : str or None
    Title for the legend. Default: "-log10(p-value)".
ax : plt.Axes or None
    If provided, use this Axes for plotting instead of creating a new one.
    If None, a new Axes will be created.
figsize : tuple[float, float], default=(8, 6)
    Size of the figure in inches.
show : bool or None, default=True
    Whether to display the figure interactively.
save : str or bool or None
    If str, path to save the image. If True, uses default name.
return_fig : bool, default=False
    Whether to return the matplotlib Figure and Axes.

Returns
-------
VolcanoPlot or tuple[Figure, Axes] or None

## `dx.ds`

### `dx.ds.get_gene_sets`

```python
get_gene_sets(collection: str = 'all', url: str | None = None, filepath: str | None = None, geneset_key: str = 'geneset', genesymbol_key: str = 'genesymbol', min_genes: int = 5, max_genes: int = 500) -> dict[str, list[str]]
```

Load and return gene sets as a dictionary.

Parameters
----------
collection : str
    Name of the collection to load. Default is "all".
url : str, optional
    URL to load the GMT file from. If None, uses the default collection.
filepath : str, optional
    Local file path to load the GMT file from. If None, uses the default collection.
geneset_key : str
    Column name for the gene set name in the output dictionary.
genesymbol_key : str
    Column name for the gene symbol in the output dictionary.
min_genes : int
    Minimum number of genes in a gene set to include. Default is 5.
max_genes : int
    Maximum number of genes in a gene set to include. Default is 500.

### `dx.ds.load_gmt`

```python
load_gmt(collection: str | None = None, url: str | None = None, filepath: str | None = None, geneset_key: str = 'geneset', genesymbol_key: str = 'genesymbol', min_genes: int = 5, max_genes: int = 500) -> pandas.core.frame.DataFrame
```

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

### `dx.ds.synthetic_adata`

```python
synthetic_adata(n_cells=3000, n_genes=2000, n_cell_types=3, n_de_genes=100, fc_range=(1.1, 3.0), mean_counts=10, dispersion=0.5, dropout_rate=0.5, n_samples=6, random_seed=42)
```

Create a synthetic AnnData object with simulated differential expression using negative binomial distribution.

Parameters
----------
n_cells : int
    Number of cells to simulate
n_genes : int
    Number of genes to simulate
n_cell_types : int
    Number of different cell types
n_de_genes : int
    Number of genes that are differentially expressed
fc_range : float
    Max fold change for differentially expressed genes
mean_counts : float
    Mean number of counts per gene
dispersion : float
    Dispersion parameter for negative binomial (higher = more variance)
dropout_rate : float
    Base dropout rate (probability of a count becoming zero)
n_samples : int
    Number of independent samples per condition
random_seed : int
    Random seed for reproducibility

Returns
-------
adata : AnnData
    Synthetic AnnData object with simulated count data and log-normalized data
