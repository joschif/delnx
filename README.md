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


# delnx

**delnx** (`/dɪˈlɒnɪks/`) is a python package for differential expression analysis of single-cell RNA sequencing data.

## Installation

### PyPI

```
pip install delnx
```

### Development version

```bash
pip install git+https://github.com/joschif/delnx.git@main
```


# Quickstart

```python
import delnx as dx

# Run differential expression analysis
results = dx.tl.de(
    adata,)
    condition_key="condition",
    group_key="cell_type",
    mode="all_vs_ref",
    reference="control",
    method="negbinom",
    backend="jax
)
```

## Documentation

For more information, check out the [documentation][documentation] and the [API reference][api documentation].


[issue tracker]: https://github.com/joschif/delnx/issues
[tests]: https://github.com/joschif/delnx/actions/workflows/test.yaml
[documentation]: https://delnx.readthedocs.io
[changelog]: https://delnx.readthedocs.io/en/latest/changelog.html
[api documentation]: https://delnx.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/delnx
[codecov]: https://codecov.io/gh/joschif/delnx
[pre-commit.ci]: https://results.pre-commit.ci/latest/github/joschif/delnx/main
