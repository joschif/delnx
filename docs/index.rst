🌳 delnx
=========

.. image:: https://img.shields.io/pypi/v/delnx.svg?color=blue
   :target: https://pypi.org/project/delnx
   :alt: PyPI version

.. image:: https://github.com/joschif/delnx/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/joschif/delnx/actions/workflows/test.yaml
   :alt: Tests

.. image:: https://codecov.io/gh/joschif/delnx/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/joschif/delnx
   :alt: Codecov

.. image:: https://results.pre-commit.ci/badge/github/joschif/delnx/main.svg
   :target: https://results.pre-commit.ci/latest/github/joschif/delnx/main
   :alt: pre-commit.ci status

.. image:: https://img.shields.io/readthedocs/delnx
   :target: https://delnx.readthedocs.io
   :alt: Documentation Status


:mod:`delnx` (``/dɪˈlɒnɪks/ | "de-lo-nix"``) is a python package for differential expression analysis of (single-cell) genomics data. It enables scalable analyses of atlas-level datasets through GPU-accelerated regression models and statistical tests implemented in `JAX <https://docs.jax.dev/en/latest/>`_. It also provides a consistent interface to perform DE analysis with other methods, such as `statsmodels <https://www.statsmodels.org/stable/index.html>`_ and `PyDESeq2 <https://pydeseq2.readthedocs.io/en/stable/>`_.

🚀 Installation
---------------

PyPI
~~~~

.. code-block:: bash

   pip install delnx

Development version
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install git+https://github.com/joschif/delnx.git@main

⚡ Quickstart
----------------

.. code-block:: python

   import delnx as dx

   # Compute size factors
   adata = dx.pp.size_factors(adata, method="ratio")

   # Estimate dispersion parameters
   adata = dx.pp.dispersion(
      adata,
      size_factor_key="size_factors",
      covariate_keys=["condition"]
   )

   # Run differential expression analysis
   results = dx.tl.de(
       adata,
       condition_key="condition",
       group_key="cell_type",
       mode="all_vs_ref",
       reference="control",
       method="negbinom",
       size_factor_key="size_factors",
       dispersion_key="dispersions",
   )

💎 Features
------------

- **Pseudobulking**: Perform DE on large multi-sample datasets by using pseudobulk aggregation.
- **Size factor estimation**: Compute size factors for normalization and DE analysis.
- **Dispersion estimation**: Estimate dispersion parameters for negative binomial models.
- **Differential expression analysis**: Consistent interface to perform DE analysis using various methods, including:

  - **Negative binomial regression** with dispersion estimates.
  - **Logistic regression** with a likelihood ratio test.
  - **ANOVA** tests based on linear models.
  - **DESeq2** through `PyDESeq2 <https://pydeseq2.readthedocs.io/en/stable/>`_, a widely used method for DE analysis of RNA-seq data.

- **GPU acceleration**: Most methods are implemented in JAX, enabling GPU acceleration for scalable DE-analysis on large datasets.

⚙️ Backends
-----------

**delnx** implements DE tests using regression models and statistical tests from various backends:

- `JAX <https://docs.jax.dev/en/latest/>`_
- `statsmodels <https://www.statsmodels.org/stable/index.html>`_
- `cuML <https://github.com/rapidsai/cuml>`_
- `PyDESeq2 <https://pydeseq2.readthedocs.io/en/stable/>`_

🗺️ Roadmap
-----------

- ✅ Provide a common interface to standard GLM-based DE tests (inspired by `Seurat::FindMarkers <https://satijalab.org/seurat/reference/findmarkers>`_)

  - ✅ Logistic regression and likelihood ratio test

    - ✅ statsmodels
    - ✅ JAX
    - ✅ cuML

  - ✅ Negative binomial regression

    - ✅ statsmodels
    - ✅ JAX

  - ✅ ANOVA

    - ✅ statsmodels
    - ✅ JAX

  - ✅ Binomial regression for binary data

    - ✅ statsmodels
    - ⬜ JAX

- ✅ Implement DESeq2 wrapper using `PyDESeq2 <https://pydeseq2.readthedocs.io/en/stable/>`_
- ✅ Implement size factor estimation methods
- ✅ Add dispersion estimation methods

  - ✅ Basic gene-wise dispersion estimation
  - ✅ DESeq2 and edgeR-inspired dispersion estimation with shrinkage
  - ✅ Take design and covariates into account for dispersion estimation

- ⬜ Plotting functions to visualize DE results
- ⬜ Gene set enrichment analysis for DE results


.. toctree::
    :maxdepth: 3
    :hidden:

    installation
    api
    contributing
    notebooks/index
