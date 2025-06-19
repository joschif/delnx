<img src="docs/_static/images/delnx.png" width="300" alt="delnx">


[![PyPI version][badge-pypi]][pypi]
[![Tests][badge-tests]][tests]
[![Codecov][badge-coverage]][codecov]
[![pre-commit.ci status][badge-pre-commit]][pre-commit.ci]
[![Documentation Status][badge-docs]][documentation]


[badge-tests]: https://github.com/joschif/delnx/actions/workflows/test.yaml/badge.svg
[badge-docs]: https://img.shields.io/readthedocs/delnx
[badge-coverage]: https://codecov.io/gh/joschif/delnx/branch/main/graph/badge.svg
[badge-pre-commit]: https://results.pre-commit.ci/badge/github/joschif/delnx/main.svg
[badge-pypi]: https://img.shields.io/pypi/v/delnx.svg?color=blue


# 🌳 delnx

**delnx** ("delonix"  / `/dɪˈlɒnɪks/`) is a python package for differential expression analysis of (single-cell) genomics data. It enables scalable analyses of atlas-level datasets through GPU-accelerated regression models and statistical tests implemented in [JAX](https://docs.jax.dev/en/latest/). It also provides a consistent interface to perform DE analysis with other methods, such as [statsmodels](https://www.statsmodels.org/stable/index.html) and [PyDESeq2](https://pydeseq2.readthedocs.io/en/stable/).

## 🚀 Installation

### PyPI

```
pip install delnx
```

### Development version

```bash
pip install git+https://github.com/joschif/delnx.git@main
```


## ⚡ Quickstart

```python
import delnx as dx

# Compute size factors
adata = dx.pp.size_factors(adata, method="ratio")

# Estimate dispersion parameters
adata = dx.pp.dispersion(adata, size_factor_key="size_factor", method="deseq2")

# Run differential expression analysis
results = dx.tl.de(
    adata,
    condition_key="condition",
    group_key="cell_type",
    mode="all_vs_ref",
    reference="control",
    method="negbinom",
    size_factor_key="size_factor",
    dispersion_key="dispersion",
)
```

## 💎 Features
- **Pseudobulking**: Perform DE on large multi-sample datasets by using pseudobulk aggregation.
- **Size factor estimation**: Compute size factors for normalization and DE analysis.
- **Dispersion estimation**: Estimate dispersion parameters for negative binomial models.
- **Differential expression analysis**: Consistent interface to perform DE analysis using various methods, including:
  - **Negative binomial regression** with dispersion estimates.
  - **Logistic regression** with a likelihood ratio test.
  - **ANOVA** tests based on linear models.
  - **DESeq2** through [PyDESeq2](https://pydeseq2.readthedocs.io/en/stable/), a widely used method for DE analysis of RNA-seq data.
- **GPU acceleration**: Most methods are implemented in JAX, enabling GPU acceleration for scalable DE-analysis on large datasets.


## ⚙️ Backends
**delnx** implements DE tests using regression models and statistical tests from various backends:

- [JAX](https://docs.jax.dev/en/latest/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [cuML](https://rapids.ai/cuml.html)
- [PyDESeq2](https://pydeseq2.readthedocs.io/en/stable/)


## 🗺️ Roadmap
- [x] Provide a common interface to standard GLM-based DE tests (inspired by [Seurat::FindMarkers](https://satijalab.org/seurat/reference/findmarkers))
    - [x] Logistic regression and likelihood ratio test
        - [x] statsmodels
        - [x] JAX
        - [x] cuML
    - [x] Negative binomial regression
        - [x] statsmodels
        - [x] JAX
    - [x] ANOVA
        - [x] statsmodels
        - [x] JAX
    - [x] Binomial regression for binary data
        - [x] statsmodels
        - [ ] JAX
- [x] Implement DESeq2 wrapper using [PyDESeq2](https://pydeseq2.readthedocs.io/en/stable/)
- [x] Implement size factor estimation methods
- [x] Add dispersion estimation methods
    - [x] Basic gene-wise dispersion estimation
    - [x] DESeq2 and edgeR-inspired dispersion estimation with shrinkage
    - [ ] Take design and covariates into account for dispersion estimation
- [ ] Plotting functions to visualize DE results
- [ ] Gene set enrichment analysis for DE results


## 📖 Documentation

For more information, check out the [documentation][documentation] and the [API reference][api documentation].



[issue tracker]: https://github.com/joschif/delnx/issues
[tests]: https://github.com/joschif/delnx/actions/workflows/test.yaml
[documentation]: https://delnx.readthedocs.io
[changelog]: https://delnx.readthedocs.io/en/latest/changelog.html
[api documentation]: https://delnx.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/delnx
[codecov]: https://codecov.io/gh/joschif/delnx
[pre-commit.ci]: https://results.pre-commit.ci/latest/github/joschif/delnx/main
