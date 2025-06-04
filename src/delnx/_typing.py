from typing import Literal

Method = Literal[
    "deseq2",
    "negbinom",
    "negbinom_jax",
    "lr",
    "lr_jax",
    "lr_cuml",
    "anova",
    "anova_jax",
    "anova_residual",
    "anova_residual_jax",
    "binomial",
]
Mode = Literal["sum", "mean", "median"]
DataType = Literal["counts", "lognorm", "binary", "auto"]
ComparisonMode = Literal["all_vs_ref", "all_vs_all", "1_vs_1"]
Backends = Literal["jax", "cuml", "statsmodels"]
